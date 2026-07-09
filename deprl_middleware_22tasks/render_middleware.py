from __future__ import annotations

import argparse
from pathlib import Path
import sys
import random

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


def render_frame(env, width: int, height: int, camera_id: int | None):
    base = env
    seen: set[int] = set()
    while hasattr(base, "env") and id(base) not in seen:
        seen.add(id(base))
        base = base.env
    base = base.unwrapped if hasattr(base, "unwrapped") else base
    for call in (
        lambda: base.render(),
        lambda: base.render(mode="rgb_array"),
        lambda: base.render(width=width, height=height, camera_id=camera_id),
    ):
        try:
            frame = call()
        except Exception:
            continue
        if frame is not None:
            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                return arr[:, :, :3].astype(np.uint8)
    sim = getattr(base, "sim", None)
    if sim is not None and hasattr(sim, "render"):
        return np.asarray(sim.render(width=width, height=height, camera_id=camera_id))[:, :, :3].astype(np.uint8)
    raise RuntimeError("Could not render an RGB frame.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a depRL policy through the middleware wrapper.")
    parser.add_argument("--env", default="MSKBenchReach-v0")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/middleware_render.mp4"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--decoder-path", default=None)
    parser.add_argument("--strict-weights", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-id", type=int, default=None)
    args = parser.parse_args(argv)

    import deprl
    import gymnasium as gym
    import imageio
    import msk_bench  # noqa: F401
    import deprl_middleware_22tasks.registry as registry

    seed = args.seed if args.seed is not None else random.randint(0, 100000)
    spec = registry.spec_for_env(args.env)
    env = gym.make(
        spec.middleware_env_id,
        render_mode="rgb_array",
        mode=args.mode or spec.mode,
        latent_dim=args.latent_dim,
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        strict_weights=args.strict_weights,
    )
    policy = deprl.load(args.checkpoint_dir, env)
    obs = reset_env(env, seed)
    frames = []
    total_reward = 0.0
    for step in range(args.steps):
        action = np.asarray(policy(obs), dtype=np.float32)
        obs, reward, done, info = step_env(env, action)
        total_reward += float(reward)
        frames.append(render_frame(env, args.width, args.height, args.camera_id))
        if (step + 1) % 100 == 0:
            print(f"step={step + 1} reward={total_reward:.2f}")
        if done:
            break
    if hasattr(env, "close"):
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps, macro_block_size=1)
    print(f"saved video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
