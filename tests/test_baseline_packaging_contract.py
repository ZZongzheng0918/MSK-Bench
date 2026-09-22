from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_top_level_distribution_packages_repository_owned_baselines() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    discovery = config["tool"]["setuptools"]["packages"]["find"]
    roots = set(discovery["where"])
    includes = set(discovery["include"])
    package_data = config["tool"]["setuptools"]["package-data"]

    assert {
        ".",
        "rl_paradigms/depRL",
        "rl_paradigms/msgym",
        "rl_paradigms/deprl_middleware_22tasks",
        "rl_paradigms/musclemimic",
    } <= roots
    assert {
        "rl_paradigms*",
        "deprl*",
        "msgym*",
        "deprl_middleware_22tasks*",
        "musclemimic*",
        "loco_mujoco*",
    } <= includes
    assert package_data["loco_mujoco"] == ["**/*.yaml"]
