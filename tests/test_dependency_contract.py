from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_release_does_not_track_generated_experiment_results() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "experiments/gpt6_direct_control/results/**"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert tracked == []


def test_public_release_metadata_and_documentation() -> None:
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    experiment_readme = (ROOT / "experiments/gpt6_direct_control/README.md").read_text(encoding="utf-8")
    license_notice = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    validation = (ROOT / "docs/validation.md").read_text(encoding="utf-8")
    user_guide = (ROOT / "USER_GUIDE.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    project = project_config()["project"]

    assert "recorded rollouts" not in root_readme
    assert "experiment code and execution instructions only" in root_readme
    assert "mock --task stairs" in experiment_readme
    assert "requires both restricted Walk and Run motion files" in experiment_readme
    assert "bare JSON array containing exactly 354" in experiment_readme
    assert "No LLM client is included" in experiment_readme
    assert "Apache License" in license_notice[:120]
    assert "Version 2.0" in license_notice[:160]
    assert "MSK-Bench" in notice
    assert "Third-party" in notice
    assert "anonymous" not in root_readme.lower()
    assert "anonymous" not in experiment_readme.lower()
    assert "peer review" not in user_guide.lower()
    assert "205 passed, 5 skipped" in validation
    assert "`test_minimal_layout.py`" in user_guide
    assert "`test_third_party_attribution.py`" in user_guide
    assert "No external source checkout is required" in user_guide
    assert "given-names: Zongzheng" in citation
    assert "family-names: Zhang" in citation
    assert project["readme"] == "README.md"
    assert project["license"] == "Apache-2.0"
    assert project["urls"]["Repository"] == "https://github.com/ZZongzheng0918/MSK-Bench"


def test_review_snapshot_scaffolding_is_not_part_of_public_release() -> None:
    assert not (ROOT / "REVIEW_FILES.json").exists()
    assert not (ROOT / "docs/review").exists()
    assert not (ROOT / "tools/prepare_review_motions.py").exists()
    assert (ROOT / "docs/data-and-licenses.md").is_file()
    assert (ROOT / "docs/restricted-motions.json").is_file()
    assert (ROOT / "tools/prepare_authorized_motions.py").is_file()


def project_config() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_core_runtime_dependencies_are_declared() -> None:
    dependencies = "\n".join(project_config()["project"]["dependencies"]).lower()

    for name in (
        "gymnasium",
        "mujoco==3.4.0",
        "dm-control==1.0.36",
        "numpy",
        "scipy",
        "scikit-video",
        "flatten-dict",
        "gitpython",
        "termcolor",
        "packaging",
        "imageio",
        "imageio-ffmpeg",
    ):
        assert name in dependencies


def test_public_baseline_extras_are_declared() -> None:
    extras = project_config()["project"]["optional-dependencies"]

    assert {
        "sb3",
        "deprl",
        "dynsyn",
        "middleware",
        "residual",
        "all",
        "dev",
    } <= set(extras)
    residual = "\n".join(extras["residual"]).lower()
    for dependency in (
        "mujoco-mjx==3.4.0",
        "numpy==2.2.6",
        "warp-lang==1.10.0",
        "optax>=0.2.4",
        "orbax==0.1.9",
        "distrax==0.1.7",
        "hydra-core",
        "ml-collections>=1.1.0",
        "opencv-python",
        "huggingface-hub",
        "seaborn>=0.13.2",
        "metrx",
        "wandb>=0.19.10",
        "viser>=1.0.10",
        "pillow>=10.0.0",
    ):
        assert dependency in residual


def test_runtime_sources_do_not_reference_external_checkouts() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    unix_home = "/" + "home/"
    forbidden = re.compile(
        rf"(?i)([a-z]:[\\/]+humanoid-bench|[a-z]:[\\/]+users[\\/]|{re.escape(unix_home)}[^/\s]+)"
    )
    violations: list[str] = []

    for relative in tracked:
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".toml", ".yaml", ".yml", ".json"}:
            continue
        if forbidden.search(path.read_text(encoding="utf-8", errors="replace")):
            violations.append(relative)

    assert violations == []
