import torch
import torch.nn as nn
import numpy as np

class SynergyModelWrapper(nn.Module):
    def __init__(self, base_model, num_synergies=64, num_muscles=416, muscle_names=None):
        super().__init__()
        self.__dict__['base_model'] = base_model
        self.num_synergies = num_synergies
        self.num_muscles = num_muscles
        
        # 定义 64 -> 416 的解冻协同层
        self.synergy_layer = nn.Linear(num_synergies, num_muscles, bias=False)
        self._init_synergy(num_synergies, num_muscles, muscle_names)

    def _init_synergy(self, num_syn, num_mus, names):
        # 初始状态下，将 416 块肌肉均匀分配到 64 个协同组
        init_mat = np.zeros((num_mus, num_syn), dtype=np.float32)
        step = num_mus // num_syn
        for i in range(num_syn):
            init_mat[i*step : (i+1)*step, i] = 1.0
        self.synergy_layer.weight.data.copy_(torch.from_numpy(init_mat))

    def initialize(self, observation_space, action_space):
        """
        核心拦截：
        1. 让 base_model 按真实的 416 维空间初始化（这样 Critic 维度就对了）。
        2. 随后劫持 Actor 的输出层，强行改为 64 维。
        """
        # 步骤 1: 正常初始化 (416-D)
        self.base_model.initialize(observation_space, action_space)

        # 步骤 2: 劫持 Actor 的输出层 (416 -> 64)
        found = False
        actor_net = self.base_model.actor
        for name, module in actor_net.named_modules():
            # 寻找输出维度为 416 的线性层
            if isinstance(module, nn.Linear) and module.out_features == action_space.shape[0]:
                new_layer = nn.Linear(module.in_features, self.num_synergies).to(module.weight.device)
                # 动态替换
                path = name.split('.')
                parent = actor_net
                for p in path[:-1]: parent = getattr(parent, p)
                setattr(parent, path[-1], new_layer)
                found = True
        
        if not found:
            print("WARNING: Synergy Hijack failed to find Actor head.")

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @property
    def actor(self):
        return SynergyActorModule(self.base_model.actor, self.synergy_layer)

    def __getattr__(self, name):
        if name in ['base_model', 'synergy_layer', 'actor', 'initialize']:
            return super().__getattr__(name)
        return getattr(self.base_model, name)

class SynergyActorModule(nn.Module):
    def __init__(self, base_actor, synergy_layer):
        super().__init__()
        self.base_actor = base_actor
        self.synergy_layer = synergy_layer

    def forward(self, obs):
        # 1. 劫持后的基础 Actor 输出 64 维分布
        dist = self.base_actor(obs)
        
        # 2. 映射均值到肌肉空间 (64 -> 416)
        muscle_mean = torch.sigmoid(self.synergy_layer(dist.mean))
        
        # 3. 映射并对齐标准差
        raw_std = dist.normal.stddev if hasattr(dist, 'normal') else dist.stddev
        avg_std = raw_std.mean(dim=-1, keepdim=True).expand_as(muscle_mean)
        
        # 4. 包装并返回 Tonic 兼容的 416 维分布
        if hasattr(dist, 'normal'):
            return type(dist)(torch.distributions.Normal(muscle_mean, avg_std))
        return torch.distributions.Normal(muscle_mean, avg_std)