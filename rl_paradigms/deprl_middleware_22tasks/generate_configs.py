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


def config_text(task: Task, encoder_path: str | None, decoder_path: str | None,
                *, latent_dim: int = 64, strict_weights: bool = False) -> str:
    if latent_dim not in (16, 32, 64, 128):
        raise ValueError("latent_dim must be 16, 32, 64, or 128")
    if strict_weights and (not encoder_path or not decoder_path):
        raise ValueError("Strict learned-latent evaluation requires both encoder and decoder")
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
  environment: deprl.environments.Gym('{task.middleware_env_id}', scaled_actions=False, latent_dim={latent_dim}, mode='{task.mode}', encoder_path={encoder_expr}, decoder_path={decoder_expr}, strict_weights={strict_weights})
  environment_name: {task.environment_name}
  full_save: 0
  header: |
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
    parser.add_argument("--latent-dim", type=int, choices=(16, 32, 64, 128), default=64)
    parser.add_argument("--strict-weights", action="store_true",
                        help="Require trained encoder/decoder (recommended for quantitative comparisons)")
    for flag in ("steps", "epoch-steps", "save-steps", "parallel", "sequential"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--working-dir", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    if args.strict_weights:
        for path in (args.encoder_path, args.decoder_path):
            if not path or not Path(path).is_file():
                parser.error("--strict-weights requires existing --encoder-path and --decoder-path")
    for key in ("steps", "epoch_steps", "save_steps", "parallel", "sequential"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        path = args.output_dir / f"msk_bench_{task.slug}_middleware.yaml"
        content = config_text(task, args.encoder_path, args.decoder_path,
                              latent_dim=args.latent_dim, strict_weights=args.strict_weights)
        overrides = {key: getattr(args, key) for key in ("steps", "epoch_steps", "save_steps")
                     if getattr(args, key) is not None}
        if overrides or args.parallel or args.sequential or args.working_dir or args.cpu:
            import yaml

            config = yaml.safe_load(content)
            if overrides:
                config["trainer_args"] = overrides
            for key in ("parallel", "sequential"):
                if getattr(args, key) is not None:
                    config["tonic"][key] = getattr(args, key)
            if args.working_dir is not None:
                config["working_dir"] = str(args.working_dir.resolve())
            if args.cpu:
                config["tonic"]["cpu_override"] = True
            content = yaml.safe_dump(config, sort_keys=False)
        path.write_text(content, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
