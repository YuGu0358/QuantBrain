from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alpha_miner.modules.asset_manifest import (
    DEFAULT_PROFILE_NAME,
    get_asset_profile,
    load_asset_manifest,
)


def offline_probe(profile_name: str, manifest_path: Path | None = None) -> dict[str, Any]:
    """Return the local manifest profile without using BRAIN credentials."""
    manifest = load_asset_manifest(manifest_path)
    get_asset_profile(profile_name, manifest=manifest)
    payload = manifest.to_dict()
    payload["probe"] = {
        "mode": "offline_manifest_export",
        "profile": profile_name,
        "live_probe": False,
    }
    return payload


def live_probe(*_: Any, **__: Any) -> dict[str, Any]:
    raise NotImplementedError("Live BRAIN dataset probing is not implemented in this offline manifest tool.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export locally verified BRAIN asset manifest profiles.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME, help="Manifest profile to export.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest YAML path.")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format.")
    parser.add_argument("--output", type=Path, default=None, help="Write output to this path instead of stdout.")
    args = parser.parse_args(argv)

    payload = offline_probe(args.profile, args.manifest)
    text = _dump_payload(payload, args.format)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _dump_payload(payload: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        import yaml

        return yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    except ModuleNotFoundError:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    main()
