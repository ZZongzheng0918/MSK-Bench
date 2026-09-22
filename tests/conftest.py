"""Only explicitly unavailable licensed-data cases are skipped in the source release."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = {
    "MSKBenchResidualWalk-v0": "walking_medium09_poses.npz",
    "MSKBenchResidualRun-v0": "walking_run04_poses.npz",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name not in {"test_environment_runtime.py", "test_pretrained_runtime.py"}:
            continue
        params = getattr(getattr(item, "callspec", None), "params", {})
        filename = RESTRICTED.get(params.get("env_id"))
        if filename and not (ROOT / "rl_paradigms/residualrl" / filename).is_file():
            item.add_marker(pytest.mark.skip(
                reason=f"Restricted motion {filename} excluded from source release; see README.md for authorized acquisition."
            ))
