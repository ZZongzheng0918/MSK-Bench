from __future__ import annotations

import torch
import torch.nn as nn


class MuscleAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x = residual + self.attn(x, x, x)[0]
        residual = x
        x = self.ln2(x)
        return residual + self.mlp(x)


class FullAnatomicalTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_muscles: int,
        num_joints: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.num_muscles = int(num_muscles)
        self.num_joints = int(num_joints)
        self.latent_dim = int(latent_dim)

        self.phys_encoder = nn.Linear(2, 64)
        self.state_encoder = nn.Linear(4, 64)
        self.joint_encoder = nn.Linear(self.num_joints, 128)
        self.cmd_projection = nn.Linear(self.latent_dim, embed_dim)
        self.input_mix = nn.Linear(64 + 64 + 128 + embed_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [MuscleAttentionBlock(embed_dim, num_heads=num_heads) for _ in range(num_blocks)]
        )
        self.final_layer = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(
        self,
        latent_action: torch.Tensor,
        priors: torch.Tensor,
        states: torch.Tensor,
        moments: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = states.shape[0]
        muscles = self.num_muscles
        joints = self.num_joints

        if priors.dim() == 2:
            priors = priors.unsqueeze(0).expand(batch_size, -1, -1)
        if moments.numel() == batch_size * muscles * joints:
            moments = moments.view(batch_size, muscles, joints)

        phys_features = self.phys_encoder(priors)
        state_features = self.state_encoder(states)
        joint_features = self.joint_encoder(moments)
        cmd_features = self.cmd_projection(latent_action).unsqueeze(1).expand(-1, muscles, -1)

        x = self.input_mix(torch.cat([phys_features, state_features, joint_features, cmd_features], dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.final_layer(x).squeeze(-1)


def build_encoder(num_muscles: int, latent_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(num_muscles, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Linear(128, latent_dim),
        nn.Tanh(),
    )
