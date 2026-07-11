import numpy as np
import pytest

from musclemimic.integrations import msk_bench


def test_checkpoint_source_precedence(monkeypatch):
    monkeypatch.setenv(msk_bench.CHECKPOINT_ENV_VAR, "/env/checkpoint")
    assert msk_bench.resolve_checkpoint_source("/argument/checkpoint") == "/argument/checkpoint"
    assert msk_bench.resolve_checkpoint_source(None) == "/env/checkpoint"
    monkeypatch.delenv(msk_bench.CHECKPOINT_ENV_VAR)
    assert msk_bench.resolve_checkpoint_source(None) == msk_bench.DEFAULT_CHECKPOINT_SOURCE


def test_blank_checkpoint_values_fall_back(monkeypatch):
    monkeypatch.setenv(msk_bench.CHECKPOINT_ENV_VAR, "   ")
    assert msk_bench.resolve_checkpoint_source("  ") == msk_bench.DEFAULT_CHECKPOINT_SOURCE


def test_policy_input_dimension_is_validated():
    with pytest.raises(ValueError, match=r"expected observation dimension 4, got 3"):
        msk_bench.validate_policy_observation(np.zeros(3, dtype=np.float32), 4)


def test_policy_action_dimension_is_validated():
    with pytest.raises(ValueError, match=r"expected action dimension 2, got 3"):
        msk_bench.validate_policy_action(np.zeros(3, dtype=np.float32), 2)
