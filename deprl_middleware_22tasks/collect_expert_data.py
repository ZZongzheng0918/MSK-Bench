from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

def _bootstrap_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in (repo_root / "MSK-Bench", repo_root / "depRL", repo_root / "deprl_middleware_22tasks"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

_bootstrap_repo_paths()

def reset_env(env, seed: int | None = None):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    return result


def build_env(args):
    import gymnasium as gym
    import msk_bench  # noqa: F401
    import deprl_middleware_22tasks.registry as registry

    spec = registry.spec_for_env(args.env)
    env_id = spec.middleware_env_id if args.middleware else spec.env_id
    kwargs = {"render_mode": None}
    if args.middleware:
        kwargs.update(
            mode=args.mode or spec.mode,
            latent_dim=args.latent_dim,
            encoder_path=args.encoder_path,
            decoder_path=args.decoder_path,
            strict_weights=args.strict_weights,
        )
    return gym.make(env_id, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect depRL expert data for MSK-Bench middleware teacher training.")
    parser.add_argument("--env", default="MSKBenchReach-v0")
    parser.add_argument("--checkpoint-dir", required=True, help="depRL run/checkpoint directory accepted by deprl.load.")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/expert_synergy.pt"))
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--middleware", action="store_true", help="Collect through the middleware env instead of the raw base env.")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--decoder-path", default=None)
    parser.add_argument("--strict-weights", action="store_true")
    args = parser.parse_args(argv)

    import deprl
    from deprl_middleware_22tasks.expert_data import extract_env_data, save_expert_dataset

    rng = np.random.default_rng(args.seed)
    env = build_env(args)
    policy = deprl.load(args.checkpoint_dir, env)
    obs = reset_env(env, args.seed)

    priors, states, moments, actions = [], [], [], []
    collected = 0
    while collected < args.samples:
        expert_action = np.asarray(policy(obs), dtype=np.float32)
        p, s, m, a = extract_env_data(env, expert_action)
        priors.append(p)
        states.append(s)
        moments.append(m)
        actions.append(a)

        noisy_action = expert_action
        if args.noise_std > 0:
            noisy_action = np.clip(
                expert_action + rng.normal(0.0, args.noise_std, size=expert_action.shape),
                -1.0,
                1.0,
            )
        obs, reward, done, info = step_env(env, noisy_action)
        collected += 1
        if done:
            obs = reset_env(env, args.seed + collected)
        if collected % 5000 == 0:
            print(f"collected {collected}/{args.samples}")

    if hasattr(env, "close"):
        env.close()
    path = save_expert_dataset(args.output, priors, states, moments, actions)
    print(f"saved expert dataset: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
