from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import PACKAGE_ROOT
from .config_loader import load_yaml


DEFAULT_PROFILE_NAME = "equity_usa_top3000"
DEFAULT_MANIFEST_PATH = PACKAGE_ROOT / "config" / "asset_manifest.yaml"


@dataclass(frozen=True)
class AssetProfile:
    name: str
    settings: dict[str, Any]
    verified_fields: tuple[str, ...]
    verified_operators: tuple[str, ...]
    group_fields: tuple[str, ...]
    status: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": dict(self.settings),
            "status": self.status,
            "provenance": dict(self.provenance),
            "verified_fields": list(self.verified_fields),
            "verified_operators": list(self.verified_operators),
            "group_fields": list(self.group_fields),
        }


@dataclass(frozen=True)
class AssetManifest:
    default_profile: str
    profiles: dict[str, AssetProfile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_profile": self.default_profile,
            "profiles": {
                name: profile.to_dict()
                for name, profile in sorted(self.profiles.items())
            },
        }


def load_asset_manifest(path: Path | None = None) -> AssetManifest:
    """Load local/offline verified BRAIN asset profiles.

    The manifest is intentionally offline-only today. It records what this repo
    treats as verified for generation and validation; it does not claim that a
    live BRAIN dataset probe has run.
    """
    manifest_path = path or DEFAULT_MANIFEST_PATH
    raw = load_yaml(manifest_path)
    default_profile = str(raw.get("default_profile") or DEFAULT_PROFILE_NAME)
    raw_profiles = raw.get("profiles") or {}
    profiles: dict[str, AssetProfile] = {}
    for name, value in raw_profiles.items():
        profile = value or {}
        profiles[str(name)] = AssetProfile(
            name=str(name),
            settings=dict(profile.get("settings") or {}),
            verified_fields=tuple(str(item) for item in profile.get("verified_fields") or []),
            verified_operators=tuple(str(item) for item in profile.get("verified_operators") or []),
            group_fields=tuple(str(item) for item in profile.get("group_fields") or []),
            status=str(profile.get("status") or "unknown"),
            provenance=dict(profile.get("provenance") or {}),
        )
    if default_profile not in profiles:
        raise ValueError(f"Default asset profile {default_profile!r} is not defined in {manifest_path}.")
    return AssetManifest(default_profile=default_profile, profiles=profiles)


def get_asset_profile(
    profile_name: str | None = None,
    *,
    manifest: AssetManifest | None = None,
    path: Path | None = None,
) -> AssetProfile:
    loaded = manifest or load_asset_manifest(path)
    name = profile_name or loaded.default_profile
    try:
        return loaded.profiles[name]
    except KeyError as exc:
        available = ", ".join(sorted(loaded.profiles))
        raise ValueError(f"Unknown asset profile {name!r}. Available profiles: {available}.") from exc


def manifest_to_dict(manifest: AssetManifest | None = None) -> dict[str, Any]:
    return (manifest or load_asset_manifest()).to_dict()


def profile_to_dict(profile: AssetProfile) -> dict[str, Any]:
    return profile.to_dict()


__all__ = [
    "AssetManifest",
    "AssetProfile",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_PROFILE_NAME",
    "get_asset_profile",
    "load_asset_manifest",
    "manifest_to_dict",
    "profile_to_dict",
]
