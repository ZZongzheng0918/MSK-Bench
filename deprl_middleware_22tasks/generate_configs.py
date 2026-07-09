from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    env_id: str
    slug: str
    mode: str

    @property
    def middleware_env_id(self) -> str:
        return self.env_id.replace("-v0", "-Middleware-v0")

    @property
    def tonic_name(self) -> str:
        return self.env_id.removesuffix("-v0") + "_Middleware_DEP"

    @property
    def environment_name(self) -> str:
        return "deprl_middleware_" + self.slug


TASKS = (
    Task("MSKBenchStand-v0", "stand", "hard"),
    Task("MSKBenchPowerlift-v0", "powerlift", "hard"),
    Task("MSKBenchSingleLegStand-v0", "single_leg_stand", "hard"),
    Task("MSKBenchSit-v0", "sit", "hard"),
    Task("MSKBenchBalance-v0", "balance", "residual"),
    Task("MSKBenchSquat-v0", "squat", "hard"),
    Task("MSKBenchWalk-v0", "walk", "hard"),
    Task("MSKBenchCrawl-v0", "crawl", "hard"),
    Task("MSKBenchRun-v0", "run", "hard"),
    Task("MSKBenchJump-v0", "jump", "hard"),
    Task("MSKBenchWalkTurn-v0", "walk_turn", "hard"),
    Task("MSKBenchSidestep-v0", "sidestep", "hard"),
    Task("MSKBenchStairs-v0", "stairs", "hard"),
    Task("MSKBenchHurdle-v0", "hurdle", "hard"),
    Task("MSKBenchStepStones-v0", "step_stones", "hard"),
    Task("MSKBenchSlide-v0", "slide", "hard"),
    Task("MSKBenchDoorOpen-v0", "door_open", "residual"),
    Task("MSKBenchReach-v0", "reach", "primate_bimanual"),
    Task("MSKBenchWalkAndSit-v0", "walk_and_sit", "hard"),
    Task("MSKBenchChinUp-v0", "chin_up", "residual"),
    Task("MSKBenchCatch-v0", "catch", "primate_bimanual"),
    Task("MSKBenchPoleWalk-v0", "pole_walk", "hard"),
)


def config_text(task: Task, encoder_path: str | None, decoder_path: str | None) -> str:
    encoder_expr = repr(encoder_path) if encoder_path else "None"
    decoder_expr = repr(decoder_path) if decoder_path else "None"
    return f"""DEP:
  bias_rate: 0.002
  buffer_size: 200
  intervention_length: 5
  intervention_proba: 0.001
  kappa: 1169.7
  normalization: independent
  q_norm_selector: l2
  regularization: 32
  s4avg: 2
  sensor_delay: 1
  tau: 40
  test_episode_every: 5
  time_dist: 5
  with_learning: true

env_args: {{}}

mpo_args:
  hidden_size: 1024
  lr_actor: 5.0e-05
  lr_critic: 8.0e-05
  lr_dual: 0.002

tonic:
  after_training: ''
  agent: deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(replay=deprl.replays.buffers.Buffer(return_steps=3, batch_size=256, steps_between_batches=1000, batch_iterations=30, steps_before_batches=1e5))
  before_training: ''
  checkpoint: last
  environment: deprl.environments.Gym('{task.middleware_env_id}', scaled_actions=False, latent_dim=64, mode='{task.mode}', encoder_path={encoder_expr}, decoder_path={decoder_expr}, strict_weights=False)
  environment_name: {task.environment_name}
  full_save: 0
  header: |
    import sys
    sys.path.insert(0, r'D:/MSK-Bench/deprl_middleware_22tasks')
    sys.path.insert(0, r'D:/MSK-Bench/MSK-Bench')
    import deprl
    import msk_bench
    import deprl_middleware_22tasks.registry
  name: {task.tonic_name}
  parallel: 4
  resume: 0
  seed: 0
  sequential: 4
  test_environment: null
  trainer: deprl.custom_trainer.Trainer(steps=int(1e8), epoch_steps=int(2e5), save_steps=int(2e6))

working_dir: ./baselines_MSKBench_Middleware
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate depRL YAML configs for 22 middleware MSK-Bench tasks.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "configs")
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--decoder-path", default=None)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        path = args.output_dir / f"msk_bench_{task.slug}_middleware.yaml"
        path.write_text(config_text(task, args.encoder_path, args.decoder_path), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
