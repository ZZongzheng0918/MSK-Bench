from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    task_name: str
    family: str
    horizon: int
    success_metric: str
    entry_point: str
    model_file: str
    observation_notes: tuple[str, ...] = ()
    kwargs: dict | None = None

    @property
    def slug(self) -> str:
        name = self.env_id.removeprefix("MSKBench").removesuffix("-v0")
        chars: list[str] = []
        for index, char in enumerate(name):
            if char.isupper() and index > 0:
                chars.append("_")
            chars.append(char.lower())
        return "".join(chars)


MSK_BENCH_ENTRY = "msk_bench.envs.msk.benchmark.msk_bench_v0"


def _task(task_name: str, class_name: str, model_file: str, family: str, horizon: int, success_metric: str, **kwargs) -> TaskSpec:
    return TaskSpec(
        env_id=f"MSKBench{task_name}-v0",
        task_name=task_name,
        family=family,
        horizon=horizon,
        success_metric=success_metric,
        entry_point=f"{MSK_BENCH_ENTRY}:{class_name}",
        model_file=model_file,
        kwargs=dict(kwargs),
    )


CANONICAL_TASKS: tuple[TaskSpec, ...] = (
    _task("Stand", "MSKBenchStandEnvV0", "full_body.xml", "posture", 1000, "solved", reset_type="init"),
    _task("Powerlift", "MSKBenchPowerliftEnvV0", "powerlift.xml", "manipulation", 1000, "hold_bonus", reset_type="init", min_height=0.5),
    _task("SingleLegStand", "MSKBenchSingleLegStandEnvV0", "full_body.xml", "posture", 1000, "solved", reset_type="init"),
    _task("Sit", "MSKBenchSitEnvV0", "sit.xml", "posture", 1000, "solved", reset_type="init", min_height=0.35, chair_pos=[-0.6, 0.0, 0.45]),
    _task("Balance", "MSKBenchBalanceEnvV0", "balance.xml", "balance", 1000, "solved", reset_type="init", min_height=0.8, max_rot=0.8),
    _task("Squat", "MSKBenchSquatEnvV0", "squat.xml", "locomotion", 1000, "solved", reset_type="init", min_height=0.35, target_x_vel=0.0, target_y_vel=0.0),
    _task("Walk", "MSKBenchWalkEnvV0", "full_body.xml", "locomotion", 1000, "solved", reset_type="init", min_height=0.8, max_rot=0.8, hip_period=100, target_x_vel=0.0, target_y_vel=1.2, target_rot=None),
    _task("Crawl", "MSKBenchCrawlEnvV0", "full_body.xml", "locomotion", 1000, "solved", reset_type="init", min_height=0.1, target_x_vel=0.5),
    _task("Run", "MSKBenchRunEnvV0", "full_body.xml", "locomotion", 1000, "solved", reset_type="none", min_height=0.75, target_x_vel=5.0, max_episode_steps=1000),
    _task("Jump", "MSKBenchJumpEnvV0", "full_body.xml", "locomotion", 1000, "solved", reset_type="init"),
    _task("WalkTurn", "MSKBenchWalkTurnEnvV0", "walk_turn.xml", "locomotion", 2000, "solved", reset_type="init", min_height=0.8),
    _task("Sidestep", "MSKBenchSidestepEnvV0", "full_body.xml", "locomotion", 1000, "solved", reset_type="init", target_y_vel=1.0, max_episode_steps=1000),
    _task("Stairs", "MSKBenchStairsEnvV0", "stairs.xml", "locomotion", 2000, "solved", max_episode_steps=2000, reset_type="init"),
    _task("Hurdle", "MSKBenchHurdleEnvV0", "hurdle.xml", "locomotion", 1000, "solved", reset_type="init", min_height=0.7, target_x_vel=1.5, max_episode_steps=1000),
    _task("StepStones", "MSKBenchStepStonesEnvV0", "step_stones.xml", "locomotion", 1000, "solved", reset_type="init", min_height=0.6, target_x_vel=0.5),
    _task("Slide", "MSKBenchSlideEnvV0", "slide.xml", "locomotion", 1500, "solved", reset_type="init", min_height=0.65, target_x_vel=1.0),
    _task("DoorOpen", "MSKBenchDoorOpenEnvV0", "door_open.xml", "manipulation", 1000, "solved", max_episode_steps=1000, reset_type="init", min_height=0.8, target_x_vel=1.0),
    _task("Reach", "MSKBenchReachEnvV0", "reach.xml", "manipulation", 500, "solved", max_episode_steps=500, reset_type="init", min_height=0.8),
    _task("WalkAndSit", "MSKBenchWalkAndSitEnvV0", "sit.xml", "locomotion", 500, "solved", reset_type="init", min_height=0.35, chair_pos=[-0.5, 0.0, 0.45]),
    _task("ChinUp", "MSKBenchChinUpEnvV0", "chin_up.xml", "manipulation", 500, "solved", max_episode_steps=500, reset_type="init", min_height=0.5),
    _task("Catch", "MSKBenchCatchEnvV0", "catch.xml", "manipulation", 200, "solved", reset_type="init", min_height=0.75, target_x_vel=0.0),
    _task("PoleWalk", "MSKBenchPoleWalkEnvV0", "pole_walk.xml", "locomotion", 1000, "solved", reset_type="init", min_height=0.6, target_x_vel=0.5),
)

CANONICAL_ENV_IDS: tuple[str, ...] = tuple(task.env_id for task in CANONICAL_TASKS)
CANONICAL_TASK_BY_ID: dict[str, TaskSpec] = {task.env_id: task for task in CANONICAL_TASKS}
CANONICAL_TASK_BY_NAME: dict[str, TaskSpec] = {task.task_name: task for task in CANONICAL_TASKS}

TASK_FAMILY_NAMES: tuple[str, ...] = ("stabilization", "locomotion", "interaction")
TASK_FAMILY_TASK_NAMES: dict[str, tuple[str, ...]] = {
    "stabilization": ("Stand", "Powerlift", "SingleLegStand", "Sit", "Balance", "Squat"),
    "locomotion": ("Walk", "Crawl", "Run", "Jump", "WalkTurn", "Sidestep"),
    "interaction": (
        "Stairs",
        "Hurdle",
        "StepStones",
        "Slide",
        "DoorOpen",
        "Reach",
        "WalkAndSit",
        "ChinUp",
        "Catch",
        "PoleWalk",
    ),
}

TASK_FAMILIES: dict[str, tuple[TaskSpec, ...]] = {
    family: tuple(CANONICAL_TASK_BY_NAME[name] for name in names)
    for family, names in TASK_FAMILY_TASK_NAMES.items()
}
TASK_FAMILY_BY_ENV_ID: dict[str, str] = {
    task.env_id: family
    for family, tasks in TASK_FAMILIES.items()
    for task in tasks
}


def tasks_in_family(family: str) -> tuple[TaskSpec, ...]:
    try:
        return TASK_FAMILIES[family]
    except KeyError as exc:
        valid = ", ".join((*TASK_FAMILY_NAMES, "all"))
        raise KeyError(f"Unknown MSK-Bench task family {family!r}. Expected one of: {valid}") from exc


def suite_for_env_id(env_id: str) -> str:
    try:
        return TASK_FAMILY_BY_ENV_ID[env_id]
    except KeyError as exc:
        raise KeyError(f"Unknown canonical MSK-Bench env id: {env_id}") from exc


def model_path_for(task: TaskSpec, body_dir: str | Path) -> str:
    return str(Path(body_dir) / task.model_file)


__all__ = [
    "CANONICAL_ENV_IDS",
    "CANONICAL_TASKS",
    "CANONICAL_TASK_BY_ID",
    "CANONICAL_TASK_BY_NAME",
    "MSK_BENCH_ENTRY",
    "TASK_FAMILIES",
    "TASK_FAMILY_BY_ENV_ID",
    "TASK_FAMILY_NAMES",
    "TASK_FAMILY_TASK_NAMES",
    "TaskSpec",
    "model_path_for",
    "suite_for_env_id",
    "tasks_in_family",
]

