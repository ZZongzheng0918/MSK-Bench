"""
Evaluation script for trained models using Stable-Baselines3.
This script loads a trained model and evaluates its performance by recording videos
of the agent's behavior in the environment.

Usage:
    python eval.py --log_path PATH_TO_LOG_DICT --model_path PATH_TO_MODEL_FILE --num_episodes NUM_EPISODES
"""

import argparse
import json
import os
from typing import Any
import sb3_contrib
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from DynSyn import SAC_DynSyn
from utils import create_env, record_video

AGENT_REGISTRY = {
    "SAC": sb3.SAC,
    "PPO": sb3.PPO,
    "TD3": sb3.TD3,
    "DDPG": sb3.DDPG,
    "A2C": sb3.A2C,
    "DQN": sb3.DQN,
    "SAC_DynSyn": SAC_DynSyn,
}
AGENT_REGISTRY.update({name: getattr(sb3_contrib, name) for name in ("ARS", "CrossQ", "MaskablePPO", "QRDQN", "RecurrentPPO", "TQC", "TRPO") if hasattr(sb3_contrib, name)})
POLICY_REGISTRY = {"MlpPolicy": "MlpPolicy", "CnnPolicy": "CnnPolicy", "MultiInputPolicy": "MultiInputPolicy"}


def _ensure_env_registered(env_name: str) -> None:
    if isinstance(env_name, str) and env_name.startswith("msgym/"):
        import msgym  # noqa: F401
def load_policy(args: Any) -> Any:
    policy = args.agent_kwargs.pop("policy", None)
    policy = "MlpPolicy" if policy is None else policy
    if policy not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy '{policy}'. Available policies: {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[policy]

def evaluate(args: argparse.Namespace) -> None:
    """Load a trained model and record evaluation videos."""
    # Load configuration from json file
    json_files = [f for f in os.listdir(args.log_path) if f.endswith(".json")]
    if len(json_files) > 1:
        raise ValueError(f"Multiple JSON files found in {args.log_path}. Expected only one.")
    elif len(json_files) == 0:
        raise ValueError(f"No JSON files found in {args.log_path}")
    config = json.load(open(os.path.join(args.log_path, json_files[0])))
        
    # Setup directories
    checkpoint_dir = os.path.join(args.log_path, "checkpoint")
    video_dir = os.path.join(args.log_path, "eval_video")
    os.makedirs(video_dir, exist_ok=True)

    # Prepare arguments
    arg_config = argparse.Namespace(**config)

    _ensure_env_registered(arg_config.env_name)

    # Load the appropriate agent class
    Agent = AGENT_REGISTRY.get(arg_config.agent)
    if Agent is None:
        raise ValueError(f"Unknown agent '{arg_config.agent}'. Available agents: {sorted(AGENT_REGISTRY)}")
    
    # Load the model
    print(f"Loading model from {checkpoint_dir}")
    if args.model_path:
        print(f"Loading specific model from {args.model_path}")
        model = Agent.load(args.model_path, **args.load_kwargs if hasattr(args, "load_kwargs") else {})
    else:
        print(f"Loading best model from {checkpoint_dir}")
        model = Agent.load(os.path.join(checkpoint_dir, "best_model.zip"), **args.load_kwargs if hasattr(args, "load_kwargs") else {})

    print(f"Model's timesteps: {model.num_timesteps}")

    # Create and normalize environment
    env = create_env(
        arg_config.env_name,
        arg_config.single_env_kwargs,
        arg_config.wrapper_list,
        render_mode="rgb_array",
    )
    vec_norm_path = os.path.join(checkpoint_dir, "best_env.zip")
    vec_norm = VecNormalize.load(vec_norm_path, DummyVecEnv([lambda: env]))

    # Record evaluation videos
    record_video(
        vec_norm,
        model,
        arg_config,
        video_dir=video_dir,
        video_ep_num=args.num_episodes,
        name_prefix=arg_config.env_name.split("/")[-1],
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model and record videos")
    parser.add_argument(
        "--log_path", "-f", type=str, default=None,
        help="Path to the log directory containing the config file",
    )
    parser.add_argument(
        "--model_path", "-m", type=str, default=None,
        help="Path to a specific model to evaluate",
    )
    parser.add_argument(
        "--num_episodes", "-n", type=int, default=3,
        help="Number of episodes to evaluate",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
