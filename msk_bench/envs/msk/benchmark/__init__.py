"""Register the canonical MSK-Bench full-body benchmark tasks."""

from __future__ import annotations

from pathlib import Path

from msk_bench.registry import CANONICAL_TASKS, model_path_for
from msk_bench.utils import gym

register = gym.register
curr_dir = Path(__file__).resolve().parent
MSK_BENCH_BODY_DIR = (curr_dir / "../../../simhive/msk_sim/body").resolve()
RESIDUAL_RL_DIR = curr_dir / "residualrl"


def msk_bench_model(filename):
    """Resolve a model filename from the bundled MSK-Bench body model directory."""
    return model_path_for(type("Task", (), {"model_file": filename})(), MSK_BENCH_BODY_DIR)


def residualrl_resource(filename):
    """Resolve a residual-control resource from the bundled residualrl folder."""
    return str(RESIDUAL_RL_DIR / filename)


def register_msk_bench_task(task_name, class_name, model_file, max_episode_steps=1000, **kwargs):
    """Register one canonical MSK-Bench task."""
    register(
        id=f"MSKBench{task_name}-v0",
        entry_point=f"msk_bench.envs.msk.benchmark.msk_bench_v0:{class_name}",
        max_episode_steps=max_episode_steps,
        kwargs={"model_path": msk_bench_model(model_file), **kwargs},
    )


def register_residual_msk_bench_task(task_name, entry_point, max_episode_steps=1000, **kwargs):
    """Register one optional residual-control MSK-Bench extension task."""
    register(
        id=f"MSKBench{task_name}-v0",
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs,
    )


for task in CANONICAL_TASKS:
    kwargs = dict(task.kwargs or {})
    max_episode_steps = int(kwargs.pop("max_episode_steps", task.horizon))
    register(
        id=task.env_id,
        entry_point=task.entry_point,
        max_episode_steps=max_episode_steps,
        kwargs={"model_path": model_path_for(task, MSK_BENCH_BODY_DIR), **kwargs},
    )

register_residual_msk_bench_task(
    "ResidualRun",
    "msk_bench.envs.msk.benchmark.residualrl.run:make_env",
    max_episode_steps=5000,
    motion_path=residualrl_resource("walking_run04_poses.npz"),
)
register_residual_msk_bench_task(
    "ResidualStair",
    "msk_bench.envs.msk.benchmark.residualrl.stair:make_env",
    max_episode_steps=2000,
    motion_path=residualrl_resource("stair_prior_89d.npz"),
)
register_residual_msk_bench_task(
    "ResidualWalk",
    "msk_bench.envs.msk.benchmark.residualrl.walk:make_env",
    max_episode_steps=1000,
    motion_path=residualrl_resource("walking_medium09_poses.npz"),
)

register(
    id="MSKBenchAgenticWalk-v0",
    entry_point="msk_bench.envs.msk.benchmark.agentic_walk_v0:MSKBenchAgenticWalkEnvV0",
    max_episode_steps=1000,
    kwargs={"model_path": msk_bench_model("full_body.xml")},
)