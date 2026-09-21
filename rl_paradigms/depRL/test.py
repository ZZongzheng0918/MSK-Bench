import imageio
import gymnasium as gym
import msk_bench  # noqa: F401


def save_gif(frames, filename="msk_bench_test.gif"):
    """Save rendered frames as a GIF."""
    print(f"Saving GIF to {filename} ...")
    imageio.mimsave(filename, frames, fps=30)
    print("GIF saved successfully.")


def run_test():
    env_name = "MSKBenchPowerlift-v0"

    try:
        print(f"Loading environment: {env_name} ...")
        env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=100)
    except Exception as exc:
        print(f"Environment failed to load: {exc}")
        print("Check that MSK-Bench registration ran and the XML model path is valid.")
        return

    try:
        obs, info = env.reset()
        print("Environment reset successfully.")
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
    except Exception as exc:
        print(f"Reset failed: {exc}")
        env.close()
        return

    frames = []
    total_reward = 0.0
    print("Running a 100-step smoke rollout...")

    for step in range(100):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            dense = info.get("rwd_dense", reward)
            print(f"Step {step}: reward={reward:.4f}, dense={dense:.4f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"Total reward: {total_reward:.4f}")

    if frames:
        save_gif(frames, f"{env_name}_test.gif")
    else:
        print("No rendered frames were captured.")

    env.close()


if __name__ == "__main__":
    run_test()
