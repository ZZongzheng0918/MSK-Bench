"""MSK-Bench boundary for the vendored MuscleMimic integration."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def third_party_root() -> Path:
    return repository_root() / "third_party" / "musclemimic"


def is_available() -> bool:
    root = third_party_root()
    return (root / "musclemimic").is_dir() and (root / "loco_mujoco").is_dir()


def provenance_status() -> str:
    return "bundled-third-party-source" if is_available() else "missing-third-party-source"