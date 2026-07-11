"""Tests for PPO config merging with backward compatibility."""

import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.ppo.config import PPO_PARAMS, get_ppo_config


class TestGetPpoConfig:
    """Tests for get_ppo_config function."""

    def test_base_config_with_ppo_config_section(self):
        """New-style config: ppo_config values should be accessible at top level."""
        config = OmegaConf.create({
            "lr": 4e-5,
            "num_envs": 8192,
            "ppo_config": {
                "num_steps": 8,
                "num_minibatches": 8,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
            },
        })

        merged = get_ppo_config(config)

        # PPO params should be accessible at top level
        assert merged.num_steps == 8
        assert merged.num_minibatches == 8
        assert merged.gamma == 0.99
        assert merged.clip_eps == 0.2

        # Shared params should still be there
        assert merged.lr == 4e-5
        assert merged.num_envs == 8192

    def test_variant_override_takes_precedence(self):
        """Top-level overrides should take precedence over ppo_config values."""
        config = OmegaConf.create({
            "lr": 4e-5,
            "num_minibatches": 128,  # Override!
            "gamma": 0.95,  # Override!
            "ppo_config": {
                "num_steps": 8,
                "num_minibatches": 8,  # Should be ignored
                "gamma": 0.99,  # Should be ignored
                "clip_eps": 0.2,
            },
        })

        merged = get_ppo_config(config)

        # Overridden values should come from top level
        assert merged.num_minibatches == 128
        assert merged.gamma == 0.95

        # Non-overridden values should come from ppo_config
        assert merged.num_steps == 8
        assert merged.clip_eps == 0.2

    def test_old_style_config_without_ppo_config(self):
        """Old-style config (no ppo_config section) should work unchanged."""
        config = OmegaConf.create({
            "lr": 4e-5,
            "num_envs": 8192,
            "num_steps": 16,
            "num_minibatches": 16,
            "gamma": 0.95,
            "clip_eps": 0.3,
        })

        merged = get_ppo_config(config)

        # Should return config as-is
        assert merged.num_steps == 16
        assert merged.num_minibatches == 16
        assert merged.gamma == 0.95
        assert merged.clip_eps == 0.3

    def test_all_ppo_params_are_merged(self):
        """All PPO_PARAMS should be merged from ppo_config."""
        ppo_values = {
            "num_steps": 8,
            "update_epochs": 2,
            "num_minibatches": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "clip_eps_vf": 0.3,
            "init_std": 1.0,
            "learnable_std": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        }

        config = OmegaConf.create({
            "lr": 4e-5,
            "ppo_config": ppo_values,
        })

        merged = get_ppo_config(config)

        for param in PPO_PARAMS:
            assert hasattr(merged, param), f"Missing param: {param}"
            assert getattr(merged, param) == ppo_values[param]

    def test_ppo_config_section_is_preserved(self):
        """The ppo_config section should still be accessible in merged config."""
        config = OmegaConf.create({
            "lr": 4e-5,
            "ppo_config": {
                "num_steps": 8,
                "gamma": 0.99,
            },
        })

        merged = get_ppo_config(config)

        # ppo_config should still exist
        assert merged.ppo_config is not None
        assert merged.ppo_config.num_steps == 8
