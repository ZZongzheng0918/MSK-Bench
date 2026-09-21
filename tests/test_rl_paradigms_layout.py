from __future__ import annotations

import re
import json
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARADIGMS = ROOT / "rl_paradigms"

TARGETS = {
    "ppo": PARADIGMS / "ppo",
    "sac": PARADIGMS / "sac",
    "depRL": PARADIGMS / "depRL",
    "deprl_middleware_22tasks": PARADIGMS / "deprl_middleware_22tasks",
    "msgym": PARADIGMS / "msgym",
    "musclemimic": PARADIGMS / "musclemimic",
    "agentic_walk": PARADIGMS / "agentic_walk",
    "residualrl": PARADIGMS / "residualrl",
}

OLD_PATHS = (
    ROOT / "ppo",
    ROOT / "sac",
    ROOT / "depRL",
    ROOT / "deprl_middleware_22tasks",
    ROOT / "msgym",
    ROOT / "third_party" / "musclemimic",
    ROOT / "msk_bench" / "envs" / "msk" / "benchmark" / "agentic_walk_v0.py",
    ROOT / "msk_bench" / "envs" / "msk" / "benchmark" / "residualrl",
)

TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".ps1",
}


def tracked_text_files() -> list[Path]:
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    return [
        ROOT / relative
        for relative in tracked
        if relative
        and (ROOT / relative).is_file()
        and Path(relative).suffix.lower() in TEXT_SUFFIXES
    ]


def test_all_rl_paradigms_share_one_physical_parent():
    assert PARADIGMS.is_dir()
    assert all(path.is_dir() for path in TARGETS.values())
    assert not any(path.exists() for path in OLD_PATHS)
    assert (TARGETS["agentic_walk"] / "clean_walk.npy").is_file()
    motions = json.loads((ROOT / "docs/restricted-motions.json").read_text())["motions"]
    assert any(item["destination"] == "rl_paradigms/residualrl/walking_medium09_poses.npz" for item in motions)
    assert (TARGETS["musclemimic"] / "LICENSE").is_file()


def test_shared_path_model_resolves_paradigm_resources():
    from msk_bench import paths

    assert paths.REPO_ROOT == ROOT
    assert paths.RL_PARADIGMS_DIR == PARADIGMS
    assert paths.paradigm_path("ppo") == TARGETS["ppo"]
    assert paths.paradigm_path(
        "residualrl",
        "walking_medium09_poses.npz",
    ) == TARGETS["residualrl"] / "walking_medium09_poses.npz"


def test_main_distribution_packages_all_vendored_paradigms():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    discovery = config["tool"]["setuptools"]["packages"]["find"]
    roots = set(discovery["where"])
    includes = set(discovery["include"])
    package_data = config["tool"]["setuptools"]["package-data"]

    assert roots >= {".", "rl_paradigms/depRL", "rl_paradigms/msgym"}
    assert includes >= {
        "benchmark_eval*", "rl_paradigms*", "deprl*", "msgym*",
        "deprl_middleware_22tasks*", "musclemimic*", "loco_mujoco*",
    }
    assert package_data["rl_paradigms.agentic_walk"] == ["clean_walk.npy"]
    assert package_data["rl_paradigms.residualrl"] == ["*.npz", "*.xml"]
    assert package_data["loco_mujoco"] == ["**/*.yaml"]


def test_moved_benchmark_entry_points_use_real_modules():
    text = (
        ROOT / "msk_bench" / "envs" / "msk" / "benchmark" / "__init__.py"
    ).read_text(encoding="utf-8")
    expected = (
        "rl_paradigms.agentic_walk.agentic_walk_v0:MSKBenchAgenticWalkEnvV0",
        "rl_paradigms.residualrl.walk:make_env",
        "rl_paradigms.residualrl.run:make_env",
        "rl_paradigms.residualrl.stair:make_env",
    )
    assert all(entry_point in text for entry_point in expected)
    assert "msk_bench.envs.msk.benchmark.residualrl" not in text
    assert "msk_bench.envs.msk.benchmark.agentic_walk_v0" not in text


def test_unified_evaluator_templates_use_paradigm_root():
    text = (ROOT / "benchmark_eval" / "evaluate.py").read_text(encoding="utf-8")
    for relative in (
        "rl_paradigms/ppo/eval_ppo_success.py",
        "rl_paradigms/sac/eval_sac_success.py",
        "rl_paradigms/depRL/eval_deprl_success.py",
        "rl_paradigms/msgym/eval_msgym_success.py",
        "rl_paradigms/deprl_middleware_22tasks/eval_middleware_success.py",
    ):
        assert relative in text


def test_middleware_configs_derive_import_paths_from_main_package():
    config_dir = ROOT / "rl_paradigms" / "deprl_middleware_22tasks" / "configs"
    configs = sorted(config_dir.glob("msk_bench_*_middleware.yaml"))

    assert len(configs) == 22
    for config in configs:
        text = config.read_text(encoding="utf-8")
        assert "from msk_bench.paths import REPO_ROOT, RL_PARADIGMS_DIR" in text
        assert "D:" + "/MSK-Bench" not in text


def test_residualrl_training_configs_use_paradigm_relative_resources():
    config_dir = ROOT / "rl_paradigms" / "depRL" / "experiments" / "msk_bench_training_files"
    for task in ("walk", "run", "stair"):
        text = (config_dir / f"msk_bench_residual_{task}.yaml").read_text(encoding="utf-8")
        assert "rl_paradigms/residualrl/" in text
        assert "D:" + "/MSK-Bench" not in text


def test_musclemimic_retarget_accepts_portable_path_arguments():
    source = (ROOT / "rl_paradigms" / "musclemimic" / "retarget.py").read_text(encoding="utf-8")

    assert '"--amass-root"' in source
    assert '"--motion-data"' in source
    assert '"--output"' in source
    assert "Path(__file__).resolve().parent" in source
    assert "/" + "home/" not in source

def test_msgym_resolves_relative_log_root_from_its_project_directory():
    source = (ROOT / "rl_paradigms" / "msgym" / "SB3-Scripts" / "train.py").read_text(encoding="utf-8")

    assert "def resolve_project_path(" in source
    assert "arg_config.log_root_dir = str(resolve_project_path(arg_config.log_root_dir))" in source

def test_tracked_text_omits_developer_machine_paths():
    drive_root = "D:" + "/MSK-Bench"
    drive_root_backslash = "D:" + "\\MSK-Bench"
    windows_profile = re.compile(
        r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\s`'\"]+"
    )
    unix_home = re.compile(r"/" + r"home/[^/\s`'\"]+")
    violations: list[str] = []

    for path in tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            drive_root.lower() in text.lower()
            or drive_root_backslash.lower() in text.lower()
            or windows_profile.search(text)
            or unix_home.search(text)
        ):
            violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []


def test_shared_evaluation_helpers_use_package_imports():
    forbidden = (
        "import " + "msk_eval_common",
        "import " + "emg_export_common",
    )
    violations: list[str] = []

    for path in tracked_text_files():
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(statement in text for statement in forbidden):
            violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []
