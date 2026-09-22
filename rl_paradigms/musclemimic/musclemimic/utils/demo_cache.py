"""
Download pre-retargeted demo motions from HuggingFace for quick testing
without AMASS.
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__)

_BIMANUAL_ENV_NAME = "MyoBimanualArm"


def download_demo_cache(
    env_name: str = _BIMANUAL_ENV_NAME,
    motion_path: str = "KIT/3/tennis_forehand_right04_poses.npz",
    repo_id: str = "amathislab/demo_dataset",
    cache_dir: str = None,
):
    """
    Download a pre-retargeted motion cache from HuggingFace.

    This allows users to test without downloading the full AMASS dataset.

    Args:
        env_name: Environment name (e.g., "MyoBimanualArm", "MyoFullBody")
        motion_path: Relative path to motion
            (e.g., "KIT/3/tennis_forehand_right04_poses.npz")
        repo_id: HuggingFace dataset repo
        cache_dir: Custom cache directory.
            If None, uses default (~/.musclemimic/caches/AMASS)

    Returns:
        Path to the downloaded cache file
    """
    # Use custom cache directory or default
    if cache_dir is None:
        cache_base = Path.home() / ".musclemimic" / "caches" / "AMASS"
    else:
        cache_base = Path(cache_dir)
    cache_base.mkdir(parents=True, exist_ok=True)

    normalized_env_name = env_name.removeprefix("Mjx")
    local_path = cache_base / normalized_env_name / motion_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    hf_path = f"{normalized_env_name}/{motion_path}"

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            repo_type="dataset",
        )
        shutil.copy2(downloaded, local_path)
        logger.info(f"Downloaded demo motion cache: {motion_path} -> {local_path}")
        return local_path
    except Exception as e:
        logger.warning(f"Could not download from HuggingFace: {e}")
        logger.warning("You may need to download AMASS and retarget manually.")
        return None


def get_demo_motions():
    """Get list of available demo motions for quick testing"""
    return {
        _BIMANUAL_ENV_NAME: [
            "gmr/BioMotionLab_NTroje/rub039/0022_throwing_hard1_poses.npz",
            "gmr/BioMotionLab_NTroje/rub109/0021_catching_and_throwing_poses.npz",
            "gmr/KIT/3/tennis_forehand_right04_poses.npz",
            "gmr/KIT/3/wave_left09_poses.npz",
            "gmr/KIT/572/throw_left03_poses.npz",
            "gmr/s9/banana_peel_1_stageii.npz",
            "gmr/s9/doorknob_use_2_stageii.npz"
        ],
        "MyoFullBody": [
            "gmr/KIT/314/walking_medium09_poses.npz",
            "gmr/KIT/348/turn_right03_poses.npz",
            "gmr/KIT/4/WalkInCounterClockwiseCircle04_poses.npz"
        ],
    }


def setup_demo(env_name: str):
    """Download all demo motions for the given environment."""
    all_demos = get_demo_motions()
    env_name = env_name.removeprefix("Mjx")
    if env_name not in all_demos:
        raise ValueError(f"Unknown env '{env_name}'. Available: {list(all_demos.keys())}")

    motions = all_demos[env_name]
    logger.info(f"Downloading {len(motions)} {env_name} demo motions...")

    downloaded = []
    for motion_path in motions:
        file = download_demo_cache(env_name, motion_path)
        if file:
            downloaded.append(motion_path)

    logger.info(f"Downloaded {len(downloaded)}/{len(motions)}:")
    for m in downloaded:
        logger.info(f"  {m}")

    return downloaded


def setup_demo_for_bimanual():
    return setup_demo(_BIMANUAL_ENV_NAME)


def setup_demo_for_myo_fullbody():
    return setup_demo("MyoFullBody")
