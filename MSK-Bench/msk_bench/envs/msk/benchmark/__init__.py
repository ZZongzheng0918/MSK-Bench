"""Register the canonical MSK-Bench full-body benchmark tasks."""

import os

import numpy as np

from msk_bench.utils import gym

register = gym.register
curr_dir = os.path.dirname(os.path.abspath(__file__))

MSK_BENCH_ENTRY = "msk_bench.envs.msk.benchmark.msk_bench_v0"
MSK_BENCH_BODY_DIR = os.path.abspath(os.path.join(curr_dir, "../../../simhive/msk_sim/body"))
RESIDUAL_RL_DIR = os.path.join(curr_dir, "residualrl")


def msk_bench_model(filename):
    """Resolve a model filename from the bundled MSK-Bench body model directory."""
    return os.path.join(MSK_BENCH_BODY_DIR, filename)


def residualrl_resource(filename):
    """Resolve a residual-control resource from the bundled residualrl folder."""
    return os.path.join(RESIDUAL_RL_DIR, filename)


def register_msk_bench_task(task_name, class_name, model_file, max_episode_steps=1000, **kwargs):
    """Register one canonical MSK-Bench task."""
    register(
        id=f"MSKBench{task_name}-v0",
        entry_point=f"{MSK_BENCH_ENTRY}:{class_name}",
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


register_msk_bench_task("Stand", "MSKBenchStandEnvV0", "full_body.xml", reset_type="init")
register_msk_bench_task("Powerlift", "MSKBenchPowerliftEnvV0", "powerlift.xml", reset_type="init", min_height=0.5)
register_msk_bench_task("SingleLegStand", "MSKBenchSingleLegStandEnvV0", "full_body.xml", reset_type="init")
register_msk_bench_task("Sit", "MSKBenchSitEnvV0", "sit.xml", reset_type="init", min_height=0.35, chair_pos=[-0.6, 0.0, 0.45])
register_msk_bench_task("Balance", "MSKBenchBalanceEnvV0", "balance.xml", reset_type="init", min_height=0.8, max_rot=0.8)
register_msk_bench_task("Squat", "MSKBenchSquatEnvV0", "squat.xml", reset_type="init", min_height=0.35, target_x_vel=0.0, target_y_vel=0.0)
register_msk_bench_task("Walk", "MSKBenchWalkEnvV0", "full_body.xml", reset_type="init", min_height=0.8, max_rot=0.8, hip_period=100, target_x_vel=0.0, target_y_vel=1.2, target_rot=None)
register_msk_bench_task("Crawl", "MSKBenchCrawlEnvV0", "full_body.xml", reset_type="init", min_height=0.1, target_x_vel=0.5)
register_msk_bench_task("Run", "MSKBenchRunEnvV0", "full_body.xml", reset_type="none", min_height=0.75, target_x_vel=5.0, max_episode_steps=1000)
register_msk_bench_task("Jump", "MSKBenchJumpEnvV0", "full_body.xml", reset_type="init")
register_msk_bench_task("WalkTurn", "MSKBenchWalkTurnEnvV0", "walk_turn.xml", reset_type="init", min_height=0.8)
register_msk_bench_task("Sidestep", "MSKBenchSidestepEnvV0", "full_body.xml", reset_type="init", target_y_vel=1.0, max_episode_steps=1000)
register_msk_bench_task("Stairs", "MSKBenchStairsEnvV0", "stairs.xml", max_episode_steps=2000, reset_type="init")
register_msk_bench_task("Hurdle", "MSKBenchHurdleEnvV0", "hurdle.xml", reset_type="init", min_height=0.7, target_x_vel=1.5, max_episode_steps=1000)
register_msk_bench_task("StepStones", "MSKBenchStepStonesEnvV0", "step_stones.xml", reset_type="init", min_height=0.6, target_x_vel=0.5)
register_msk_bench_task("Slide", "MSKBenchSlideEnvV0", "slide.xml", reset_type="init", min_height=0.65, target_x_vel=1.0)
register_msk_bench_task("DoorOpen", "MSKBenchDoorOpenEnvV0", "door_open.xml", max_episode_steps=1000, reset_type="init", min_height=0.8, target_x_vel=1.0)
register_msk_bench_task("Reach", "MSKBenchReachEnvV0", "reach.xml", max_episode_steps=500, reset_type="init", min_height=0.8)
register_msk_bench_task("WalkAndSit", "MSKBenchWalkAndSitEnvV0", "sit.xml", reset_type="init", min_height=0.35, chair_pos=np.array([-0.5, 0.0, 0.45]))
register_msk_bench_task("ChinUp", "MSKBenchChinUpEnvV0", "chin_up.xml", max_episode_steps=500, reset_type="init", min_height=0.5)
register_msk_bench_task("Catch", "MSKBenchCatchEnvV0", "catch.xml", reset_type="init", min_height=0.75, target_x_vel=0.0)
register_msk_bench_task("PoleWalk", "MSKBenchPoleWalkEnvV0", "pole_walk.xml", reset_type="init", min_height=0.6, target_x_vel=0.5)
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
