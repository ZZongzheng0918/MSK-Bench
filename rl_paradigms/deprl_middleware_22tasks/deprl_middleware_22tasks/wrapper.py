from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .expert_data import actuator_moments, actuator_priors, muscle_states, sim_from_env
from .networks import FullAnatomicalTransformer, build_encoder


class BioMiddlewareWrapper(gym.Wrapper):
    """Transformer-based action middleware for MSK-Bench depRL policies.

    If encoder/decoder weights are not supplied, the wrapper falls back to a safe
    pass-through mode. Set strict_weights=True when you want missing weights to
    fail fast instead.
    """

    def __init__(
        self,
        env: gym.Env,
        latent_dim: int = 64,
        mode: str = "hard",
        encoder_path: str | None = None,
        decoder_path: str | None = None,
        strict_weights: bool = False,
        residual_scale: float = 1.0,
        tube_radius: float = 0.25,
        lower_body_tube_radius: float = 0.10,
        lower_body_cutoff: int = 290,
        penalty_scale: float = 5.0,
        decay_steps: int = 2_000_000,
        device: str | None = None,
    ):
        super().__init__(env)
        self.latent_dim = int(latent_dim)
        self.mode = str(mode).lower()
        self.encoder_path = str(encoder_path) if encoder_path else None
        self.decoder_path = str(decoder_path) if decoder_path else None
        self.strict_weights = bool(strict_weights)
        self.residual_scale = float(residual_scale)
        self.tube_radius = float(tube_radius)
        self.lower_body_tube_radius = float(lower_body_tube_radius)
        self.lower_body_cutoff = int(lower_body_cutoff)
        self.penalty_scale = float(penalty_scale)
        self.decay_steps = int(decay_steps)
        self.step_count = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        sim = sim_from_env(env)
        self.num_muscles = int(sim.model.nu)
        self.num_joints = int(sim.model.nv)
        self.encoder = build_encoder(self.num_muscles, self.latent_dim).to(self.device)
        self.decoder = FullAnatomicalTransformer(self.latent_dim, self.num_muscles, self.num_joints).to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        self._priors_gpu: torch.Tensor | None = None
        self.enabled = self._load_weights()

        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def _load_weights(self) -> bool:
        encoder = Path(self.encoder_path) if self.encoder_path else None
        decoder = Path(self.decoder_path) if self.decoder_path else None
        if not encoder or not decoder or not encoder.exists() or not decoder.exists():
            if self.strict_weights:
                raise FileNotFoundError(
                    f"Missing middleware weights: encoder={self.encoder_path}, decoder={self.decoder_path}"
                )
            return False
        self.encoder.load_state_dict(torch.load(encoder, map_location=self.device))
        self.decoder.load_state_dict(torch.load(decoder, map_location=self.device))
        return True

    def reset(self, **kwargs):
        self.step_count = 0
        self._priors_gpu = None
        return self.env.reset(**kwargs)

    @property
    def muscle_states(self):
        return muscle_states(sim_from_env(self))

    def merge_args(self, args: dict[str, Any] | None) -> None:
        if args:
            for key, value in args.items():
                setattr(self, key, value)

    def apply_args(self) -> None:
        pass

    def _gpu_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sim = sim_from_env(self)
        if self._priors_gpu is None:
            priors = actuator_priors(sim)
            self._priors_gpu = torch.from_numpy(priors).float().to(self.device).unsqueeze(0)
        states = torch.from_numpy(muscle_states(sim)).float().to(self.device, non_blocking=True).unsqueeze(0)
        moments = torch.from_numpy(actuator_moments(sim)).float().to(self.device, non_blocking=True).unsqueeze(0)
        return self._priors_gpu, states, moments

    def _teacher_action_01(self, action: np.ndarray) -> np.ndarray:
        if not self.enabled or self.mode in {"off", "passthrough", "none"}:
            return np.clip((action + 1.0) / 2.0, 0.0, 1.0)
        action_t = torch.from_numpy(action.astype(np.float32)).to(self.device).unsqueeze(0)
        priors, states, moments = self._gpu_inputs()
        with torch.no_grad():
            latent_intent = self.encoder(action_t)
            filtered = self.decoder(latent_intent, priors, states, moments)
        return np.clip(filtered.squeeze(0).detach().cpu().numpy(), 0.0, 1.0)

    def _decay(self) -> float:
        return max(0.01, 1.0 - (self.step_count / max(float(self.decay_steps), 1.0)))

    def _hard_action(self, action: np.ndarray, teacher_01: np.ndarray) -> tuple[np.ndarray, float]:
        action_01 = np.clip((action + 1.0) / 2.0, 0.0, 1.0)
        safe_01 = np.clip(action_01, teacher_01 - self.tube_radius, teacher_01 + self.tube_radius)
        safe_01 = np.clip(safe_01, 0.0, 1.0)
        violation = float(np.mean((action_01 - safe_01) ** 2))
        return safe_01 * 2.0 - 1.0, violation

    def _residual_action(self, action: np.ndarray, teacher_01: np.ndarray) -> tuple[np.ndarray, float]:
        teacher_env = teacher_01 * 2.0 - 1.0
        residual = action * self.residual_scale
        env_action = np.clip(teacher_env + residual, -1.0, 1.0)
        return env_action, float(np.mean(residual**2))

    def _soft_action(self, action: np.ndarray, teacher_01: np.ndarray) -> tuple[np.ndarray, float]:
        action_01 = np.clip((action + 1.0) / 2.0, 0.0, 1.0)
        return action, float(np.mean((action_01 - teacher_01) ** 2))

    def _primate_bimanual_action(self, action: np.ndarray, teacher_01: np.ndarray) -> tuple[np.ndarray, float]:
        action_01 = np.clip((action + 1.0) / 2.0, 0.0, 1.0)
        cutoff = min(max(self.lower_body_cutoff, 0), action.shape[0])
        env_action = np.array(action, copy=True)
        if cutoff > 0:
            lower_01 = np.clip(
                action_01[:cutoff],
                teacher_01[:cutoff] - self.lower_body_tube_radius,
                teacher_01[:cutoff] + self.lower_body_tube_radius,
            )
            lower_01 = np.clip(lower_01, 0.0, 1.0)
            env_action[:cutoff] = lower_01 * 2.0 - 1.0
            violation = float(np.mean((action_01[:cutoff] - lower_01) ** 2))
        else:
            violation = 0.0
        return np.clip(env_action, -1.0, 1.0), violation

    def transform_action(self, action: np.ndarray) -> tuple[np.ndarray, float, float]:
        action = np.asarray(action, dtype=np.float32)
        if not self.enabled or self.mode in {"off", "passthrough", "none"}:
            return np.clip(action, -1.0, 1.0), 0.0, 0.0
        teacher_01 = self._teacher_action_01(action)
        if self.mode == "hard":
            env_action, violation = self._hard_action(action, teacher_01)
        elif self.mode == "residual":
            env_action, violation = self._residual_action(action, teacher_01)
        elif self.mode == "soft":
            env_action, violation = self._soft_action(action, teacher_01)
        elif self.mode == "primate_bimanual":
            env_action, violation = self._primate_bimanual_action(action, teacher_01)
        else:
            raise ValueError(f"Unknown middleware mode: {self.mode}")
        decay = self._decay()
        penalty = self.penalty_scale * decay * violation if self.mode != "residual" else 0.0
        return np.clip(env_action, -1.0, 1.0), penalty, violation

    def step(self, action):
        self.step_count += 1
        env_action, penalty, violation = self.transform_action(np.asarray(action, dtype=np.float32))
        result = self.env.step(env_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        info = dict(info)
        info.update(
            middleware_enabled=self.enabled,
            middleware_mode=self.mode,
            middleware_penalty=float(penalty),
            middleware_violation_mse=float(violation),
            middleware_decay=float(self._decay()),
        )
        return obs, float(reward) - float(penalty), bool(terminated), bool(truncated), info
