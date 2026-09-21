import json
import os
from collections import deque

import gymnasium as gym
import torch

torch.set_default_dtype(torch.float32)


class DEP:
    """
    DEP Implementation from Der et al.(2015).
    PyTorch is used instead of numpy to speed up computation.
    """

    def __init__(self, params_path="default_path"):
        if params_path == "default_path":
            dirname = os.path.dirname(__file__)
            params_path = os.path.join(
                dirname, "param_files/default_agents.json"
            )

        with open(params_path, "r") as f:
            self.params = json.load(f)["DEP"]

    def initialize(self, observation_space, action_space, seed=None):
        action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_space.shape)
        )
        self.num_sensors = action_space.shape[0]
        self.num_motors = action_space.shape[0]
        self.n_env = 1

        self.act_scale = self.act_high = torch.tensor(action_space.high)

        self.action_space = action_space
        self.obs_spec = observation_space.shape
        self.set_params(self.params)

    # =========================================================
    # 👇 Step 函数：保留了之前的维度防御逻辑
    # =========================================================
    def step(self, observations, steps=None):
        # 1. 转为 Tensor
        if not isinstance(observations, torch.Tensor):
            device = self.C_norm.device if hasattr(self, 'C_norm') else 'cpu'
            observations = torch.as_tensor(observations, dtype=torch.float32, device=device)

        # 2. 维度检查
        expected_dim = self.C_norm.shape[-1] # 112
        actual_dim = observations.shape[-1]
        
        if actual_dim != expected_dim:
            if actual_dim == 416 and expected_dim == 112:
                raise RuntimeError(
                    "❌ 致命错误：Agent 收到了 416维 肌肉数据！Wrapper 透传失败！"
                )
            
            if not hasattr(self, "_logged_dim_mismatch"):
                print(f"⚠️ [DEP] 维度修正: {actual_dim} -> {expected_dim}")
                self._logged_dim_mismatch = True
            observations = observations[..., :expected_dim]

        if getattr(self, 'header', None) is not None:
             pass
            
        return self._get_action(observations).numpy(force=True)
            
    def set_params(self, param_dict):
        for k, v in param_dict.items():
            setattr(self, k, v)
        self._reset()

    def _reset(self, obs_shape=None):
        if obs_shape:
            self.n_env = [obs_shape[0] if len(obs_shape) > 1 else 1][0]
        
        # 强制 n_env 为 1 (如果我们希望多环境共享一个 DEP 策略)
        # 这样能避免 _learn_controller 中的维度困惑
        # 但为了兼容性，我们保持原逻辑，在 update 时做 reduction
        
        self.M = torch.broadcast_to(
            -torch.eye(self.num_motors, self.num_sensors),
            (self.n_env, self.num_motors, self.num_sensors),
        )
        self.C = torch.zeros((self.n_env, self.num_motors, self.num_sensors))
        self.C_norm = torch.zeros(
            (self.n_env, self.num_motors, self.num_sensors)
        )
        self.Cb = torch.zeros((self.n_env, self.num_motors))
        self.obs_smoothed = torch.zeros((self.n_env, self.num_sensors))
        self.buffer = deque(maxlen=self.buffer_size)
        self.obs_smoothed = torch.zeros(self.obs_spec)
        self.t = 0

    def _get_action(self, obs):
        if self.s4avg > 1 and self.t > 0:
            self.obs_smoothed += (obs - self.obs_smoothed) / self.s4avg
        else:
            self.obs_smoothed = obs

        self.buffer.append([self.obs_smoothed.detach().clone(), None])
        
        if self.with_learning and len(self.buffer) > (2 + self.time_dist):
            self._learn_controller()
            
        y = self._compute_action()
        self.buffer[-1][1] = y.detach().clone()
        self.t += 1
        return y

    def _q_norm(self, q):
        reg = 10.0 ** (-self.regularization)
        if self.q_norm_selector == "l2":
            q_norm = 1.0 / (torch.linalg.norm(q, axis=-1) + reg)
        elif self.q_norm_selector == "max":
            q_norm = 1.0 / (max(abs(q), axis=-1) + reg)
        elif self.q_norm_selector == "none":
            q_norm = 1.0
        else:
            raise NotImplementedError
        return q_norm

    def _compute_action(self):
        # 兼容性处理：如果 obs 是 [10, 112] 而 C_norm 是 [1, 112, 112]
        # 我们希望结果是 [10, 112]
        # einsum "ijk, ik -> ij" 会要求 i 匹配
        # 如果不匹配 (1 vs 10)，我们手动 broadcast C_norm
        
        C_use = self.C_norm
        obs_use = self.obs_smoothed
        
        if C_use.shape[0] == 1 and obs_use.shape[0] > 1:
            # 广播 C_norm 到 batch 大小
            C_use = C_use.expand(obs_use.shape[0], -1, -1)
            
        q = torch.einsum("ijk, ik->ij", C_use, obs_use)

        q = torch.einsum(
            "ij, i->ij",
            q,
            self._q_norm(q),
        )
        
        # Cb 同理广播
        Cb_use = self.Cb
        if Cb_use.shape[0] == 1 and q.shape[0] > 1:
            Cb_use = Cb_use.expand(q.shape[0], -1)

        y = torch.maximum(
            torch.tensor([-1.0]),
            torch.minimum(
                torch.tensor([1.0]), torch.tanh(q * self.kappa + Cb_use)
            ),
        )
        y = torch.einsum("ij, j->ij", y, self.act_scale)
        return y

    def _learn_controller(self):
        """
        Update DEP by one learning step.
        """
        self.C = self._compute_C()
        
        R = torch.einsum("ijk, imk->ijm", self.C, self.M)
        reg = 10.0 ** (-self.regularization)
        
        if self.normalization == "independent":
            factor = self.kappa / (torch.linalg.norm(R, axis=-1) + reg)
            self.C_norm = torch.einsum("ijk,ik->ijk", self.C, factor)
        elif self.normalization == "none":
            self.C_norm = self.C
        elif self.normalization == "global":
            norm = torch.linalg.norm(R)
            self.C_norm = self.C * self.kappa / (norm + reg)
        else:
            raise NotImplementedError

        # 👇 【修复重点】Cb 更新的维度对齐
        if self.bias_rate >= 0:
            yy = self.buffer[-2][1] # 可能是 [10, 112]
            
            # 先计算出想要的更新量
            update_delta = torch.clip(yy * self.bias_rate, -0.05, 0.05)
            
            # 如果 Cb 是 [1, 112] 但 update_delta 是 [10, 112]
            # 我们取平均值 (mean reduction)
            if self.Cb.shape[0] == 1 and update_delta.shape[0] > 1:
                update_delta = update_delta.mean(dim=0, keepdim=True)
                
            self.Cb -= (update_delta + self.Cb * 0.001)
        else:
            self.Cb *= 0

    def _compute_C(self):
        C = torch.zeros_like(self.C)
        
        for s in range(2, min(self.t - self.time_dist, self.tau)):
            x = self.buffer[-s][0][:, : self.num_sensors]
            xx = self.buffer[-s - 1][0][:, : self.num_sensors]
            
            # Buffer 取出的数据可能是 [10, 112]
            
            if self.time_dist == 0:
                 xx_t = x
            else:
                 xx_t = self.buffer[-s - self.time_dist][0][:, : self.num_sensors]
                 
            xxx_t = self.buffer[-s - 1 - self.time_dist][0][:, : self.num_sensors]

            chi = x - xx
            v = xx_t - xxx_t
            
            # 处理 mu 计算的广播: M [1, N, N], chi [10, N]
            M_use = self.M
            if M_use.shape[0] == 1 and chi.shape[0] > 1:
                M_use = M_use.expand(chi.shape[0], -1, -1)
                
            mu = torch.einsum("ijk, ik->ij", M_use, chi)

            # 计算单步更新 [10, 112, 112]
            C_update = torch.einsum("ij, ik->ijk", mu, v)
            
            # 👇 【修复重点】C 更新的维度对齐
            # 如果 C 是 [1, 112, 112] 但 C_update 是 [10, 112, 112]
            # 取平均值
            if C.shape[0] == 1 and C_update.shape[0] > 1:
                C_update = C_update.mean(dim=0, keepdim=True)
                
            C += C_update
            
        return C