from __future__ import annotations

import random
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .common import read_json, write_json


PACKAGE_ROOT = Path(__file__).parent.parent


@dataclass
class LLMProvider:
    name: str
    role: str
    model_id: str = ""
    api_base: str | None = None
    api_key_env: str = ""
    client_type: str = "openai_compat"
    win_rate: float = 0.0
    calls: int = 0
    wins: int = 0
    total_latency_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    cost_per_1k_tokens_usd: float = 0.0
    input_cost_per_1k_usd: float | None = None
    output_cost_per_1k_usd: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LLMProvider":
        return cls(
            name=str(value["name"]),
            role=str(value["role"]),
            model_id=_str_or_default(value.get("model_id")),
            api_base=_optional_str(value.get("api_base")),
            api_key_env=_str_or_default(value.get("api_key_env")),
            client_type=_str_or_default(value.get("client_type"), "openai_compat"),
            win_rate=float(value.get("win_rate", 0.0)),
            calls=int(value.get("calls", 0)),
            wins=int(value.get("wins", 0)),
            total_latency_ms=float(value.get("total_latency_ms", 0.0)),
            total_tokens_in=int(value.get("total_tokens_in", 0)),
            total_tokens_out=int(value.get("total_tokens_out", 0)),
            cost_per_1k_tokens_usd=float(value.get("cost_per_1k_tokens_usd", 0.0)),
            input_cost_per_1k_usd=_optional_float(value.get("input_cost_per_1k_usd")),
            output_cost_per_1k_usd=_optional_float(value.get("output_cost_per_1k_usd")),
        )

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        if self.input_cost_per_1k_usd is not None or self.output_cost_per_1k_usd is not None:
            input_cost = self.input_cost_per_1k_usd if self.input_cost_per_1k_usd is not None else 0.0
            output_cost = self.output_cost_per_1k_usd if self.output_cost_per_1k_usd is not None else 0.0
            return (tokens_in / 1000.0 * input_cost) + (tokens_out / 1000.0 * output_cost)
        return ((tokens_in + tokens_out) / 1000.0) * self.cost_per_1k_tokens_usd


class LLMRouter:
    def __init__(
        self,
        providers: list[LLMProvider | Mapping[str, Any]],
        epsilon: float = 0.10,
        daily_budget_usd: float | None = None,
        rng: random.Random | None = None,
    ):
        self.epsilon = float(epsilon)
        self.daily_budget_usd = daily_budget_usd
        self.spent_usd = 0.0
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._explore_offsets: dict[str, int] = {}
        self._providers: dict[tuple[str, str], LLMProvider] = {}
        self._providers_by_role: dict[str, list[LLMProvider]] = {}

        for provider in providers:
            item = provider if isinstance(provider, LLMProvider) else LLMProvider.from_mapping(provider)
            self._providers[(item.name, item.role)] = item
            self._providers_by_role.setdefault(item.role, []).append(item)

    def pick(self, role: str) -> LLMProvider:
        candidates = self._providers_by_role.get(role, [])
        if not candidates:
            raise ValueError(f"No LLM provider configured for role: {role}")
        if len(candidates) == 1:
            return candidates[0]
        # Budget guard: if >80% of daily budget used, pick cheapest provider for this role
        if self.daily_budget_usd is not None and self.daily_budget_usd > 0:
            utilization = self.spent_usd / self.daily_budget_usd
            if utilization > 0.80:
                return min(candidates, key=_budget_cost_per_1k)
        if self.epsilon >= 1.0:
            return self._cycle_exploration(role, candidates)
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return self._rng.choice(candidates)
        return max(candidates, key=lambda provider: provider.win_rate)

    def record_result(
        self,
        provider_name: str,
        role: str,
        passed: bool,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        with self._lock:
            provider = self._providers.get((provider_name, role))
            if provider is None:
                raise ValueError(f"No LLM provider configured for name={provider_name!r}, role={role!r}")

            provider.calls += 1
            provider.wins += int(passed)
            provider.win_rate = provider.wins / provider.calls
            provider.total_latency_ms += latency_ms
            provider.total_tokens_in += tokens_in
            provider.total_tokens_out += tokens_out
            self.spent_usd += provider.cost_usd(tokens_in=tokens_in, tokens_out=tokens_out)

    def budget_remaining_usd(self) -> float:
        if self.daily_budget_usd is None:
            return float("inf")
        return self.daily_budget_usd - self.spent_usd

    def get_state(self) -> dict[str, Any]:
        providers: dict[str, dict[str, dict[str, Any]]] = {}
        for provider in self._providers.values():
            providers.setdefault(provider.name, {})[provider.role] = asdict(provider)
        return {
            "daily_budget_usd": self.daily_budget_usd,
            "epsilon": self.epsilon,
            "providers": providers,
            "spent_usd": self.spent_usd,
        }

    def save_state(self, path: Path) -> None:
        write_json(path, self.get_state())

    def save(self, path: Path) -> None:
        self.save_state(path)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "LLMRouter":
        providers_path = Path(path) if path is not None else PACKAGE_ROOT / "config" / "llm_providers.yaml"
        try:
            import yaml
        except ModuleNotFoundError:
            from .config_loader import load_yaml

            config = load_yaml(providers_path)
        else:
            with providers_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

        providers: list[dict[str, Any]] = []
        for provider_name, provider_config in config.get("providers", {}).items():
            item = dict(provider_config)
            item["name"] = provider_name
            item["win_rate"] = item.pop("weight_initial", 0.0)
            item["input_cost_per_1k_usd"] = item.pop("cost_per_1k_input", None)
            item["output_cost_per_1k_usd"] = item.pop("cost_per_1k_output", None)
            item["model_id"] = _str_or_default(item.get("model_id"))
            item["api_base"] = _optional_str(item.get("api_base"))
            item["api_key_env"] = _str_or_default(item.get("api_key_env"))
            item["client_type"] = _str_or_default(item.get("client_type"), "openai_compat")
            providers.append(item)

        return cls(providers=providers, epsilon=0.10)

    @classmethod
    def load_state(cls, path: Path) -> "LLMRouter":
        state = read_json(path)
        if state is None:
            raise FileNotFoundError(path)

        providers: list[dict[str, Any]] = []
        for provider_name, roles in state.get("providers", {}).items():
            for role, provider_state in roles.items():
                item = dict(provider_state)
                item.setdefault("name", provider_name)
                item.setdefault("role", role)
                providers.append(item)

        router = cls(
            providers=providers,
            epsilon=float(state.get("epsilon", 0.10)),
            daily_budget_usd=state.get("daily_budget_usd"),
        )
        router.spent_usd = float(state.get("spent_usd", 0.0))
        return router

    @classmethod
    def from_state(cls, path: Path) -> "LLMRouter":
        return cls.load_state(path)

    @classmethod
    def load(cls, path: Path) -> "LLMRouter":
        return cls.load_state(path)

    def _cycle_exploration(self, role: str, candidates: list[LLMProvider]) -> LLMProvider:
        offset = self._explore_offsets.get(role, 0)
        self._explore_offsets[role] = offset + 1
        return candidates[offset % len(candidates)]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _budget_cost_per_1k(provider: LLMProvider) -> float:
    input_cost = (
        provider.input_cost_per_1k_usd
        if provider.input_cost_per_1k_usd is not None
        else provider.cost_per_1k_tokens_usd
    )
    output_cost = (
        provider.output_cost_per_1k_usd
        if provider.output_cost_per_1k_usd is not None
        else provider.cost_per_1k_tokens_usd
    )
    return input_cost + output_cost


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _str_or_default(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)
