import os
from typing import Any, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from utils import record_video

class SaveConfigToTensorboardCallback(BaseCallback):
    """Log hyperparameters and config to TensorBoard at training start."""

    def __init__(
        self,
        log_dir: str,
        config_str: str,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.log_dir = log_dir
        self.config_str = config_str

    def _on_training_start(self) -> None:
        self.logger.record(
            "configs",
            self.config_str.replace("\n", "<br>")
        )
        
    def add_environment_info(self, env_info: str) -> None:
        """Log environment info string to TensorBoard."""
        self.logger.record("env_info", env_info)

    def _on_step(self) -> bool:
        return True


class SaveVecNormalizeOnBestCallback(BaseCallback):
    """Save VecNormalize stats when EvalCallback finds a new best model."""

    def __init__(self, save_path: str, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            vec_norm.save(self.save_path)
            if self.verbose >= 1:
                print(f"Saved VecNormalize to {self.save_path}")
        return True

class VideoRecorderCallback(BaseCallback):
    """Record and save videos at a fixed step frequency during training."""

    def __init__(
        self,
        args: Any,
        record_freq: int,
        video_dir: str,
        video_ep_num: int,
        env_nums: int = 4,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)

        self.record_freq = record_freq
        self.video_dir = video_dir
        self.video_ep_num = video_ep_num
        self.args = args
        self.env_nums = env_nums
        self._warned_render_mode = False

    def _on_step(self) -> bool:
        # 1. record_freq <= 0 时关闭视频录制
        if self.record_freq is None or self.record_freq <= 0:
            return True

        # 2. 没到录制频率，不录
        if self.n_calls % self.record_freq != 0:
            return True

        # 3. SB3 的 VecVideoRecorder 必须要求 render_mode == "rgb_array"
        render_mode = getattr(self.training_env, "render_mode", None)

        if render_mode != "rgb_array":
            if not self._warned_render_mode:
                print(
                    f"[VideoRecorderCallback] Skip video recording because "
                    f"training_env.render_mode={render_mode}. "
                    f"VecVideoRecorder requires render_mode='rgb_array'. "
                    f"Training will continue normally."
                )
                self._warned_render_mode = True
            return True

        # 4. 尝试录制；即使录制失败，也不要中断训练
        try:
            os.makedirs(self.video_dir, exist_ok=True)

            record_video(
                self.training_env,
                self.model,
                self.args,
                self.video_dir,
                self.video_ep_num,
                name_prefix=f"{self.args.agent}-{self.num_timesteps}"
            )

        except AssertionError as e:
            print(f"[VideoRecorderCallback] Video recording skipped: {e}")
            return True

        except Exception as e:
            print(f"[VideoRecorderCallback] Video recording failed but training continues: {e}")
            return True

        return True

class TensorboardCallback(BaseCallback):
    """Log reward components from env info dict to TensorBoard."""

    def __init__(
        self,
        info_keywords: Any,
        reward_freq: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.rollout_info: dict = {}
        self.reward_freq = reward_freq
        self.n_rollout = 1
        self.info_dict: Optional[dict] = None

    def _on_rollout_start(self) -> None:
        if self.info_dict is not None:
            self.rollout_info = {key: [] for key in self.info_dict}

    def _on_step(self) -> bool:
        if self.info_dict is None:
            self.info_dict = self.locals["infos"][0]
            # remove the keys that are not int or float
            self.info_dict = {
                k: v
                for k, v in self.info_dict.items()
                if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))
            }
            self.rollout_info = {key: [] for key in self.info_dict}
        if self.reward_freq != 0 and self.n_rollout % self.reward_freq == 0:
            for key in self.info_dict.keys():
                vals = [info[key] for info in self.locals["infos"]]
                self.rollout_info[key].extend(vals)
        return True

    def _on_rollout_end(self) -> None:
        if (
            self.reward_freq != 0
            and self.n_rollout % self.reward_freq == 0
            and self.info_dict is not None
        ):
            for key in self.info_dict:
                self.logger.record(
                    "reward/" + key,
                    np.mean(self.rollout_info[key]),
                )
        self.n_rollout += 1
