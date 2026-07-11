from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf

from musclemimic.algorithms.common import checkpoint_hooks as hooks
from musclemimic.algorithms.common.checkpoint_utils import TrainingConfig, compute_resume_state
from musclemimic.algorithms.ppo import runner


def _clear_checkpoint_callback_cache() -> None:
    if hasattr(hooks.create_jax_checkpoint_host_callback, "__cached_instance__"):
        delattr(hooks.create_jax_checkpoint_host_callback, "__cached_instance__")


def _collect_schedule_counts(opt_state) -> list[int]:
    counts: list[int] = []

    def _visit(node):
        if isinstance(node, optax.ScaleByScheduleState):
            counts.append(int(node.count))
        return node

    jax.tree_util.tree_map(
        _visit,
        opt_state,
        is_leaf=lambda x: isinstance(x, optax.ScaleByScheduleState),
    )
    return counts


def test_compute_resume_state_same_config_additional_timesteps():
    """Resume with unchanged config should use straightforward remaining updates."""
    cfg = TrainingConfig(num_envs=2048, num_steps=8, num_minibatches=8, update_epochs=1)
    steps_per_update = cfg.num_envs * cfg.num_steps
    completed_updates = 200
    base_global_ts = completed_updates * steps_per_update

    total_timesteps = base_global_ts + 57 * steps_per_update
    done, remaining, changed = compute_resume_state(
        checkpoint_metadata={
            "update_number": completed_updates,
            "global_timestep": base_global_ts,
            "num_envs": cfg.num_envs,
            "num_steps": cfg.num_steps,
        },
        current_config=cfg,
        total_timesteps=total_timesteps,
    )

    assert done == completed_updates
    assert remaining == 57
    assert changed is False


def test_compute_resume_state_changed_config_uses_current_steps_per_update():
    """When num_envs/num_steps change, remaining updates must use current config."""
    ckpt_num_envs = 2048
    ckpt_num_steps = 8
    ckpt_steps_per_update = ckpt_num_envs * ckpt_num_steps
    completed_updates = 120
    base_global_ts = completed_updates * ckpt_steps_per_update

    current_cfg = TrainingConfig(num_envs=1024, num_steps=20, num_minibatches=8, update_epochs=1)
    current_steps_per_update = current_cfg.num_envs * current_cfg.num_steps

    # Choose a target that is not divisible by current_steps_per_update to verify
    # ceiling division path for remaining updates.
    total_timesteps = base_global_ts + 7 * current_steps_per_update + (current_steps_per_update // 3)

    done, remaining, changed = compute_resume_state(
        checkpoint_metadata={
            "update_number": completed_updates,
            "global_timestep": base_global_ts,
            "num_envs": ckpt_num_envs,
            "num_steps": ckpt_num_steps,
        },
        current_config=current_cfg,
        total_timesteps=total_timesteps,
    )

    assert done == completed_updates
    assert changed is True
    assert remaining == 8  # 7 full + ceil(extra/current_steps_per_update)


def test_handle_checkpointing_passes_updates_done_and_resume_baseline(monkeypatch):
    """Periodic checkpoint should pass updates_done and resume baseline into callback factory."""
    _clear_checkpoint_callback_cache()

    captured = []
    create_calls = []

    def fake_ckpt_cb(*args):
        captured.append(args)
        return 0

    def fake_create_cb(*args, **kwargs):
        create_calls.append(kwargs)
        return fake_ckpt_cb, object()

    def fake_io_callback(cb, _shape, *args):
        cb(*args)
        return jnp.int32(0)

    monkeypatch.setattr(hooks, "create_jax_checkpoint_host_callback", fake_create_cb)
    monkeypatch.setattr(runner.jax.experimental, "io_callback", fake_io_callback)

    train_state = SimpleNamespace(
        params={"w": jnp.array([1.0], dtype=jnp.float32)},
        run_stats={},
        opt_state={},
        step=jnp.int32(17),
    )
    cfg = SimpleNamespace(
        save_checkpoints=True,
        checkpoint_interval=5,
        save_checkpoints_on_validation=False,
        validation_interval=11,
    )

    runner._handle_checkpointing(  # pylint: disable=protected-access
        train_state=train_state,
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        lr=jnp.float32(3e-4),
        counter=jnp.int32(10),
        updates_done=jnp.int32(123),
        base_global_timestep=55_555,
        base_completed_updates=100,
        config=cfg,
        agent_conf=SimpleNamespace(config=OmegaConf.create({"experiment": {}})),
        env=SimpleNamespace(),
    )

    assert len(captured) == 1
    assert len(create_calls) == 1
    assert create_calls[0]["base_global_timestep"] == 55_555
    assert create_calls[0]["base_completed_updates"] == 100
    callback_args = captured[0]
    assert len(callback_args) == 7
    assert int(callback_args[3]) == 17  # train_state.step
    assert int(callback_args[4]) == 123  # updates_done


def test_save_final_checkpoint_uses_total_updates_and_resume_baseline(monkeypatch):
    """Final checkpoint should save base+remaining updates and carry resume baseline."""
    _clear_checkpoint_callback_cache()

    captured = []
    create_calls = []

    def fake_ckpt_cb(*args):
        captured.append(args)
        return 0

    def fake_create_cb(*args, **kwargs):
        create_calls.append(kwargs)
        return fake_ckpt_cb, object()

    def fake_io_callback(cb, _shape, *args):
        cb(*args)
        return jnp.int32(0)

    monkeypatch.setattr(hooks, "create_jax_checkpoint_host_callback", fake_create_cb)
    monkeypatch.setattr(runner.jax.experimental, "io_callback", fake_io_callback)

    final_train_state = SimpleNamespace(
        params={"w": jnp.array([1.0], dtype=jnp.float32)},
        run_stats={},
        opt_state={},
        step=jnp.int32(999),
    )
    runner_state = (
        final_train_state,
        None,
        None,
        jnp.array([11, 22], dtype=jnp.uint32),  # rng
        jnp.float32(1e-4),  # lr
        None,
        None,
        None,
    )

    runner._save_final_checkpoint(  # pylint: disable=protected-access
        runner_state=runner_state,
        base_global_timestep=5_000_000_000,
        base_completed_updates=80,
        remaining_updates=20,
        agent_conf=SimpleNamespace(config=OmegaConf.create({"experiment": {}})),
        env=SimpleNamespace(),
    )

    assert len(captured) == 1
    assert len(create_calls) == 1
    assert create_calls[0]["base_global_timestep"] == 5_000_000_000
    assert create_calls[0]["base_completed_updates"] == 80
    callback_args = captured[0]
    assert len(callback_args) == 7
    assert int(callback_args[3]) == 999  # train_state.step
    assert int(callback_args[4]) == 100  # base_completed + remaining


def test_checkpoint_host_callback_uses_explicit_update_and_global_timestep(monkeypatch):
    """Host callback metadata should reconstruct global timestep from resume baseline."""

    class FakeManager:
        def __init__(self):
            self.saved = []

        def save_checkpoint(self, step, agent_conf, agent_state, metadata):
            self.saved.append((step, agent_conf, agent_state, metadata))
            return f"/tmp/checkpoint_{step}"

    fake_manager = FakeManager()
    monkeypatch.setattr(hooks, "create_checkpoint_manager", lambda *args, **kwargs: fake_manager)

    class DummyAlgo:
        @staticmethod
        def _agent_state(train_state):
            return SimpleNamespace(train_state=train_state)

    exp_cfg = OmegaConf.create(
        {
            "experiment": {
                "checkpoint_dir": "/tmp/ckpts",
                "num_envs": 64,
                "num_steps": 20,
                "num_minibatches": 8,
                "update_epochs": 1,
            }
        }
    )
    agent_conf = SimpleNamespace(
        config=exp_cfg,
        network=lambda *args, **kwargs: None,
        tx=object(),
    )
    env = SimpleNamespace(mjx_backend="jax")

    base_global_timestep = 6_656_000_000
    base_completed_updates = 16_250

    host_cb, _mgr = hooks.create_jax_checkpoint_host_callback(
        DummyAlgo,
        agent_conf,
        exp_cfg,
        env,
        base_global_timestep=base_global_timestep,
        base_completed_updates=base_completed_updates,
    )
    assert _mgr is fake_manager

    host_cb(
        {"p": jnp.array([1.0], dtype=jnp.float32)},  # ts_params
        {},  # ts_run_stats
        {},  # ts_opt_state
        jnp.int32(0),  # ts_step (would imply update=0 if derived)
        jnp.int32(base_completed_updates + 50),  # updates_done
        jnp.array([1, 2], dtype=jnp.uint32),  # rng_key
        jnp.float32(3e-4),  # current_lr
    )

    assert len(fake_manager.saved) == 1
    save_step, _agent_conf, _agent_state, md = fake_manager.saved[0]

    expected_global_timestep = base_global_timestep + 50 * 64 * 20
    assert save_step == base_completed_updates + 50
    assert md.update_number == base_completed_updates + 50
    assert md.global_timestep == expected_global_timestep
    assert md.step == 0


def test_init_train_state_resume_with_lr_reset_and_std_reset():
    """Resume should reset schedule counter/step and optionally reset action std."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    # Includes ScaleByScheduleState so reset_lr_schedule_count has real effect.
    tx = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lambda count: jnp.asarray(1.0, dtype=jnp.float32)),
        optax.scale(-1.0),
    )

    params = {
        "w": jnp.array([1.0, -1.0], dtype=jnp.float32),
        "log_std": jnp.zeros((3,), dtype=jnp.float32),
    }
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for _ in range(3):
        _updates, opt_state = tx.update(grads, opt_state, params)

    loaded_counts = _collect_schedule_counts(opt_state)
    assert loaded_counts and all(c > 0 for c in loaded_counts)

    loaded_ts = runner.TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(123, dtype=jnp.int32),
        run_stats={},
    )

    resumed = runner._init_train_state(  # pylint: disable=protected-access
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        tx=tx,
        agent_state=SimpleNamespace(train_state=loaded_ts),
        config=OmegaConf.create(
            {
                "reset_lr_schedule_on_resume": True,
                "reset_std_on_resume": 0.5,
            }
        ),
    )

    assert int(resumed.step) == 0
    assert jnp.allclose(resumed.params["log_std"], jnp.log(0.5), atol=1e-6)

    resumed_counts = _collect_schedule_counts(resumed.opt_state)
    assert resumed_counts, "expected ScaleByScheduleState in optimizer state"
    assert all(c == 0 for c in resumed_counts)


def test_init_train_state_resume_without_resets_preserves_step_and_std():
    """Resume without reset flags should preserve optimizer step and action std."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    tx = optax.adam(learning_rate=3e-4)
    params = {
        "w": jnp.array([0.1], dtype=jnp.float32),
        "log_std": jnp.array([0.2, -0.3], dtype=jnp.float32),
    }
    loaded_ts = runner.TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        step=jnp.asarray(77, dtype=jnp.int32),
        run_stats={},
    )

    resumed = runner._init_train_state(  # pylint: disable=protected-access
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        tx=tx,
        agent_state=SimpleNamespace(train_state=loaded_ts),
        config=OmegaConf.create({"reset_lr_schedule_on_resume": False}),
    )

    assert int(resumed.step) == 77
    assert jnp.allclose(resumed.params["log_std"], params["log_std"])


def test_init_train_state_resume_reset_std_only_preserves_step():
    """With reset_std_on_resume only, step stays unchanged while log_std resets."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    tx = optax.adam(learning_rate=3e-4)
    params = {
        "w": jnp.array([0.3, -0.7], dtype=jnp.float32),
        "log_std": jnp.array([0.2, -0.3, 0.5], dtype=jnp.float32),
    }
    loaded_ts = runner.TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        step=jnp.asarray(314, dtype=jnp.int32),
        run_stats={},
    )

    resumed = runner._init_train_state(  # pylint: disable=protected-access
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        tx=tx,
        agent_state=SimpleNamespace(train_state=loaded_ts),
        config=OmegaConf.create(
            {
                "reset_lr_schedule_on_resume": False,
                "reset_std_on_resume": 0.25,
            }
        ),
    )

    assert int(resumed.step) == 314
    assert jnp.allclose(resumed.params["log_std"], jnp.log(0.25), atol=1e-6)
    assert jnp.allclose(resumed.params["w"], params["w"])


def test_init_train_state_local_auto_resume_skips_one_shot_resets():
    """Local auto-resume should preserve schedule count, step, and std."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    tx = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lambda count: jnp.asarray(1.0, dtype=jnp.float32)),
        optax.scale(-1.0),
    )

    params = {
        "w": jnp.array([0.4, -0.2], dtype=jnp.float32),
        "log_std": jnp.array([0.1, -0.4], dtype=jnp.float32),
    }
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for _ in range(4):
        _updates, opt_state = tx.update(grads, opt_state, params)

    loaded_counts = _collect_schedule_counts(opt_state)
    assert loaded_counts and all(c > 0 for c in loaded_counts)

    loaded_ts = runner.TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(271, dtype=jnp.int32),
        run_stats={},
    )

    resumed = runner._init_train_state(  # pylint: disable=protected-access
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        tx=tx,
        agent_state=SimpleNamespace(train_state=loaded_ts),
        config=OmegaConf.create(
            {
                "reset_lr_schedule_on_resume": True,
                "reset_std_on_resume": 0.25,
            }
        ),
        apply_resume_resets=False,
    )

    assert int(resumed.step) == 271
    assert jnp.allclose(resumed.params["log_std"], params["log_std"])
    assert _collect_schedule_counts(resumed.opt_state) == loaded_counts


def test_compute_resume_info_stored_target_used_on_auto_resume():
    """Auto-resume should use stored target_global_timestep from checkpoint."""

    config = OmegaConf.create(
        {
            "num_updates": 10,
            "total_timesteps": 320,
            "num_envs": 4,
            "num_steps": 8,
            "num_minibatches": 1,
            "update_epochs": 1,
        }
    )
    train_state = SimpleNamespace(step=jnp.asarray(3, dtype=jnp.int32))
    agent_state = SimpleNamespace(train_state=train_state)
    # Checkpoint saved during a finetune run that resolved target to 416
    resume_info = {
        "update_number": 3,
        "global_timestep": 96,
        "target_global_timestep": 416,
        "num_envs": 4,
        "num_steps": 8,
        "num_minibatches": 1,
        "update_epochs": 1,
    }

    completed, remaining, base_ts, base_step, target = runner._compute_resume_info(
        config, agent_state, resume_info, train_state,
    )

    assert completed == 3
    assert remaining == 10  # (416 - 96) / 32 = 10
    assert base_ts == 96
    assert base_step == 3
    assert target == 416


def test_compute_resume_info_first_explicit_resume_uses_additional():
    """First explicit resume (no stored target) should add total_timesteps on top."""

    config = OmegaConf.create(
        {
            "num_updates": 10,
            "total_timesteps": 320,
            "num_envs": 4,
            "num_steps": 8,
            "num_minibatches": 1,
            "update_epochs": 1,
        }
    )
    train_state = SimpleNamespace(step=jnp.asarray(3, dtype=jnp.int32))
    agent_state = SimpleNamespace(train_state=train_state)
    # External checkpoint — no target_global_timestep stored
    resume_info = {
        "update_number": 3,
        "global_timestep": 96,
        "target_global_timestep": 0,
        "num_envs": 4,
        "num_steps": 8,
        "num_minibatches": 1,
        "update_epochs": 1,
    }

    completed, remaining, base_ts, base_step, target = runner._compute_resume_info(
        config, agent_state, resume_info, train_state,
    )

    assert completed == 3
    assert remaining == 10  # (320 + 96 - 96) / 32 = 10
    assert base_ts == 96
    assert base_step == 3
    assert target == 416  # 320 + 96


def test_compute_resume_info_finetune_preempt_preserves_target():
    """Scenario D: finetune preempted mid-run, auto-resume reads stored target."""

    config = OmegaConf.create(
        {
            "num_updates": 15,  # config says 15 but stored target overrides
            "total_timesteps": 500,
            "num_envs": 4,
            "num_steps": 8,
            "num_minibatches": 1,
            "update_epochs": 1,
        }
    )
    train_state = SimpleNamespace(step=jnp.asarray(10, dtype=jnp.int32))
    agent_state = SimpleNamespace(train_state=train_state)
    # Finetune started at global_ts=1000, target was 1000+500=1500.
    # Preempted at global_ts=1200.
    resume_info = {
        "update_number": 10,
        "global_timestep": 1200,
        "target_global_timestep": 1500,
        "num_envs": 4,
        "num_steps": 8,
        "num_minibatches": 1,
        "update_epochs": 1,
    }

    completed, remaining, base_ts, base_step, target = runner._compute_resume_info(
        config, agent_state, resume_info, train_state,
    )

    assert target == 1500
    assert remaining == 10  # (1500 - 1200) / 32 = 9.375 → ceil = 10
    assert base_ts == 1200


def test_checkpoint_host_callback_via_io_callback_under_jit_no_concretization(monkeypatch):
    """JIT + io_callback path should execute host int()/float() casts without tracer errors."""

    class FakeManager:
        def __init__(self):
            self.saved = []

        def save_checkpoint(self, step, agent_conf, agent_state, metadata):
            self.saved.append((step, metadata))
            return f"/tmp/checkpoint_{step}"

    fake_manager = FakeManager()
    monkeypatch.setattr(hooks, "create_checkpoint_manager", lambda *args, **kwargs: fake_manager)

    class DummyAlgo:
        @staticmethod
        def _agent_state(train_state):
            return SimpleNamespace(train_state=train_state)

    exp_cfg = OmegaConf.create(
        {
            "experiment": {
                "checkpoint_dir": "/tmp/ckpts",
                "num_envs": 64,
                "num_steps": 20,
                "num_minibatches": 8,
                "update_epochs": 1,
            }
        }
    )
    agent_conf = SimpleNamespace(
        config=exp_cfg,
        network=lambda *args, **kwargs: None,
        tx=object(),
    )
    env = SimpleNamespace(mjx_backend="jax")

    base_global_timestep = 6_656_000_000
    base_completed_updates = 16_250
    host_cb, _mgr = hooks.create_jax_checkpoint_host_callback(
        DummyAlgo,
        agent_conf,
        exp_cfg,
        env,
        base_global_timestep=base_global_timestep,
        base_completed_updates=base_completed_updates,
    )
    assert _mgr is fake_manager

    params = {"p": jnp.array([1.0], dtype=jnp.float32)}
    run_stats = {}
    opt_state = {}
    rng_key = jnp.array([7, 9], dtype=jnp.uint32)

    @jax.jit
    def _call_host(step, updates_done, lr):
        return jax.experimental.io_callback(
            host_cb,
            jnp.int32(0),
            params,
            run_stats,
            opt_state,
            step,
            updates_done,
            rng_key,
            lr,
        )

    out = _call_host(
        jnp.int32(12),
        jnp.int32(base_completed_updates + 5),
        jnp.float32(1e-4),
    )
    jax.block_until_ready(out)

    assert len(fake_manager.saved) == 1
    save_step, md = fake_manager.saved[0]
    assert save_step == base_completed_updates + 5
    assert md.step == 12
    assert md.update_number == base_completed_updates + 5
    assert md.global_timestep == base_global_timestep + 5 * 64 * 20
