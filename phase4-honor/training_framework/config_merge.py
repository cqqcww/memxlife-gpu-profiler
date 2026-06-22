"""Compose base configs with model/data profiles and dotted overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .config import (
    ExperimentConfig,
    _parse_scalar,
    config_from_dict,
    dump_yaml_dict,
    load_yaml_dict,
)

KNOWN_CONFIG_SECTIONS = {
    "metadata",
    "model",
    "data",
    "optimizer",
    "scheduler",
    "trainer",
    "logging",
    "checkpoint",
    "agent",
}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_path(path: str | Path, project_root: Path | None = None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and project_root is not None:
        resolved = project_root / resolved
    return resolved


def profile_name(raw: dict[str, Any], fallback: Path) -> str:
    profile = raw.get("profile")
    if isinstance(profile, dict) and profile.get("name"):
        return str(profile["name"])
    metadata = raw.get("metadata")
    if isinstance(metadata, dict) and metadata.get("name"):
        return str(metadata["name"])
    return fallback.stem


def profile_notes(raw: dict[str, Any]) -> str:
    profile = raw.get("profile")
    if not isinstance(profile, dict):
        return ""
    notes = []
    if profile.get("family"):
        notes.append(f"family={profile['family']}")
    if profile.get("expected_memory"):
        notes.append(f"expected_memory={profile['expected_memory']}")
    if profile.get("expected_token_count_note"):
        notes.append(str(profile["expected_token_count_note"]))
    caveats = profile.get("caveats")
    if isinstance(caveats, list):
        notes.extend(str(item) for item in caveats)
    return " | ".join(notes)


def extract_profile_sections(raw: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for key in KNOWN_CONFIG_SECTIONS:
        if key in raw and key != "metadata":
            extracted[key] = raw[key]
    if "trainer_recommendations" in raw:
        extracted["trainer"] = raw["trainer_recommendations"]
    return extracted


def set_dotted_value(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        raise ValueError("override key must not be empty")
    target = raw
    for part in parts[:-1]:
        current = target.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
        target = current
    target[parts[-1]] = value


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str] | None = None) -> dict[str, Any]:
    merged = deepcopy(raw)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, value = override.split("=", 1)
        set_dotted_value(merged, key.strip(), _parse_scalar(value))
    return merged


def resolve_config_dict(
    base: str | Path,
    *,
    model_profile: str | Path | None = None,
    data_profile: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    base_path = resolve_path(base, project_root)
    raw = load_yaml_dict(base_path)
    metadata = dict(raw.get("metadata") or {})
    metadata["base_config"] = str(base)
    note_parts = [str(metadata.get("notes") or "").strip()]

    if model_profile:
        model_path = resolve_path(model_profile, project_root)
        model_raw = load_yaml_dict(model_path)
        raw = deep_merge(raw, extract_profile_sections(model_raw))
        metadata["model_profile"] = profile_name(model_raw, model_path)
        note = profile_notes(model_raw)
        if note:
            note_parts.append(f"model_profile: {note}")

    if data_profile:
        data_path = resolve_path(data_profile, project_root)
        data_raw = load_yaml_dict(data_path)
        raw = deep_merge(raw, extract_profile_sections(data_raw))
        metadata["data_profile"] = profile_name(data_raw, data_path)
        note = profile_notes(data_raw)
        if note:
            note_parts.append(f"data_profile: {note}")

    metadata["notes"] = " || ".join(part for part in note_parts if part)
    raw["metadata"] = metadata
    return apply_overrides(raw, overrides)


def load_composed_config(
    base: str | Path,
    *,
    model_profile: str | Path | None = None,
    data_profile: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    project_root: Path | None = None,
) -> ExperimentConfig:
    return config_from_dict(
        resolve_config_dict(
            base,
            model_profile=model_profile,
            data_profile=data_profile,
            overrides=overrides,
            project_root=project_root,
        )
    )


def write_resolved_config(raw: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    dump_yaml_dict(raw, out)
    return out
