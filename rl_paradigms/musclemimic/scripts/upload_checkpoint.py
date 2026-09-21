"""
Upload checkpoint to HuggingFace.

Usage:
    python scripts/upload_checkpoint.py /path/to/checkpoint_30000 --repo your-org/model-name
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_checkpoint(checkpoint_path: Path, repo_id: str):
    """
    Upload an Orbax checkpoint directory to HuggingFace.

    Args:
        checkpoint_path: Path to checkpoint directory (e.g., checkpoint_30000)
        repo_id: HuggingFace repo ID (e.g., "your-org/model-name")
    """
    api = HfApi()

    # Validate checkpoint structure
    required = ["train_state", "metadata", "config", "_CHECKPOINT_METADATA"]
    missing = [r for r in required if not (checkpoint_path / r).exists()]
    if missing:
        raise ValueError(f"Invalid checkpoint - missing: {missing}")

    # Create repo if needed
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repository ready: {repo_id}")
    except Exception as e:
        print(f"Warning: {e}")

    # Upload
    print(f"Uploading {checkpoint_path} to {repo_id}...")
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint directory")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repo ID")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    upload_checkpoint(args.checkpoint, args.repo)


if __name__ == "__main__":
    main()
