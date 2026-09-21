from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESIDUAL_DIR = ROOT / "rl_paradigms" / "residualrl"


def source(task: str) -> str:
    return (RESIDUAL_DIR / f"{task}.py").read_text(encoding="utf-8")


def load_pure_function(task: str, name: str):
    tree = ast.parse(source(task))
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(RESIDUAL_DIR / f"{task}.py"), "exec"), namespace)
    return namespace[name]


def load_ast_class(task: str, name: str):
    text = source(task)
    tree = ast.parse(text)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)

    def require_existing_path(path, _name):
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return str(resolved)

    namespace = {
        "np": np,
        "mujoco": mujoco,
        "gymnasium": SimpleNamespace(Wrapper=object, Env=object),
        "SimpleNamespace": SimpleNamespace,
        "_require_existing_path": require_existing_path,
    }
    exec(compile(module, str(RESIDUAL_DIR / f"{task}.py"), "exec"), namespace)
    return namespace[name]


def minimal_stair_model():
    body_xml = []
    joint_index = 0
    axes = ("1 0 0", "0 1 0", "0 0 1")
    for body_index in range(28):
        joint_count = min(3, 82 - joint_index)
        joints = "".join(
            (
                f'<joint name="hinge_{joint_index + offset}" '
                f'type="hinge" axis="{axes[offset]}"/>'
            )
            for offset in range(joint_count)
        )
        body_xml.append(
            f'<body name="link_{body_index}" pos="{0.03 * body_index} 0 0">'
            f'{joints}<geom type="sphere" size="0.01" mass="0.01"/>'
            '</body>'
        )
        joint_index += joint_count

    xml = f"""
    <mujoco>
      <worldbody>
        <body name="root" pos="0 0 1">
          <freejoint/>
          <geom type="sphere" size="0.1" mass="1"/>
          {"".join(body_xml)}
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def test_walk_preserves_supplied_safety_template_and_registry_factory():
    text = source("walk")
    assert "class NPZTrajectoryAdapter" in text
    assert "class FullBodyGoalWrapper" in text
    assert "class MSKBenchResidualWalkEnvV0" in text
    assert "class MSKBenchResidualWalkWrapper" in text
    assert "def _apply_qacc_shield(" in text
    assert "physics_failure_penalty = -50.0" in text
    assert "def make_env(**kwargs):" in text
    assert 'else "rl_paradigms.residualrl.walk"' in text
    assert "_residual_l2_weight = 0.02" in text
    assert "metabolic_weight = 0.05" in text
    assert "metabolic_smoothing = 0.1" in text


def test_stair_progress_reward_matches_paper_formula():
    reward = load_pure_function("stair", "_stair_progress_reward")
    assert reward(3.0, 2.0) == 0.3 * (2.0 + 0.5 * 1.5)
    assert reward(-1.0, -2.0) == 0.0


def test_stair_adapter_supports_bundled_states_motion():
    motion_path = RESIDUAL_DIR / "stair_prior_89d.npz"
    with np.load(motion_path, allow_pickle=False) as trajectory:
        assert "states" in trajectory.files
        states = np.asarray(trajectory["states"])

    assert states.ndim == 2
    assert states.shape[1] == 89

    text = source("stair")
    assert 'else "rl_paradigms.residualrl.stair"' in text
    tree = ast.parse(text)
    adapter = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "NPZTrajectoryAdapter"
    )
    adapter_source = ast.get_source_segment(text, adapter)
    assert adapter_source is not None
    assert "allow_pickle=False" in adapter_source
    assert 'trajectory["states"]' in adapter_source
    assert "mujoco.mj_differentiatePos" in adapter_source


def test_stair_adapter_derives_qvel_from_states_motion(tmp_path):
    adapter_class = load_ast_class("stair", "NPZTrajectoryAdapter")
    model = minimal_stair_model()
    assert model.nq == 89
    assert model.nv == 88

    dt = 0.02
    states = np.zeros((4, model.nq), dtype=np.float64)
    states[:, 3] = 1.0
    states[:, 0] = [0.0, 0.1, 0.4, 0.9]
    states[:, 1] = [0.0, -0.05, -0.1, -0.2]
    joint_delta = np.linspace(0.001, 0.082, 82)
    states[:, 7:] = np.arange(4)[:, None] * joint_delta

    motion_path = tmp_path / "states_only.npz"
    np.savez(motion_path, states=states)
    adapter = adapter_class(str(motion_path), model, dt=dt)

    assert adapter.qpos_data.shape == (4, model.nq)
    assert adapter.qvel_data.shape == (4, model.nv)
    expected_qvel = np.zeros((4, model.nv), dtype=np.float64)
    intervals = []
    for frame_index in range(4):
        start = max(frame_index - 1, 0)
        end = min(frame_index + 1, 3)
        interval = (end - start) * dt
        intervals.append(interval)
        mujoco.mj_differentiatePos(
            model,
            expected_qvel[frame_index],
            interval,
            states[start],
            states[end],
        )

    assert intervals[0] == pytest.approx(dt)
    assert intervals[-1] == pytest.approx(dt)
    assert intervals[1:-1] == pytest.approx([2.0 * dt, 2.0 * dt])
    np.testing.assert_allclose(adapter.qpos_data, states)
    np.testing.assert_allclose(adapter.qvel_data, expected_qvel)
    np.testing.assert_allclose(adapter.root_lin_vel, adapter.qvel_data[:, :3])
    np.testing.assert_allclose(adapter.root_ang_vel, adapter.qvel_data[:, 3:6])
    np.testing.assert_allclose(adapter.joint_vel, adapter.qvel_data[:, 6:])


def test_stair_adapter_rejects_malformed_states_shape(tmp_path):
    adapter_class = load_ast_class("stair", "NPZTrajectoryAdapter")
    model = minimal_stair_model()
    malformed_path = tmp_path / "malformed_states.npz"
    np.savez(malformed_path, states=np.zeros((2, 88), dtype=np.float64))

    with pytest.raises(ValueError, match="qpos mismatch"):
        adapter_class(str(malformed_path), model)


def test_stair_copied_pelvis_position_is_immutable_snapshot():
    text = source("stair")
    assert "def _copied_pelvis_position(" in text
    copied_pelvis_position = load_pure_function(
        "stair",
        "_copied_pelvis_position",
    )
    data = SimpleNamespace(
        xpos=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float64,
        ),
        qpos=np.array([7.0, 8.0, 9.0, 1.0], dtype=np.float64),
    )

    body_snapshot = copied_pelvis_position(data, 1)
    fallback_snapshot = copied_pelvis_position(data, -1)
    data.xpos[1, :] = -1.0
    data.qpos[:3] = -2.0

    np.testing.assert_array_equal(
        body_snapshot,
        np.array([4.0, 5.0, 6.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        fallback_snapshot,
        np.array([7.0, 8.0, 9.0], dtype=np.float32),
    )


def test_stair_snapshots_current_reference_before_future_lookup():
    text = source("stair")
    tree = ast.parse(text)
    wrapper = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "MSKBenchResidualStairWrapper"
    )
    wrapper_source = ast.get_source_segment(text, wrapper)
    assert wrapper_source is not None

    pelvis_lookup = wrapper_source.index("pelvis_id =")
    future_lookup = wrapper_source.index("future_ref_data =", pelvis_lookup)
    snapshot_region = wrapper_source[pelvis_lookup:future_lookup]
    assert (
        "actual_pelvis_pos = _copied_pelvis_position("
        in snapshot_region
    )
    assert (
        "current_ref_pelvis_pos = _copied_pelvis_position("
        in snapshot_region
    )
    assert snapshot_region.count("_copied_pelvis_position(") == 2

    first_snapshot = wrapper_source.index(
        "current_ref_pelvis_pos = _copied_pelvis_position("
    )
    assert first_snapshot < future_lookup
    progress_start = wrapper_source.index(
        "progress_direction =",
        future_lookup,
    )
    future_region = wrapper_source[future_lookup:progress_start]
    assert (
        "future_pelvis_pos = _copied_pelvis_position("
        in future_region
    )

    root_dev_start = wrapper_source.index("root_dev = float(", future_lookup)
    root_dev_end = wrapper_source.index(
        "residual_terminated =",
        root_dev_start,
    )
    root_dev_source = wrapper_source[root_dev_start:root_dev_end]
    assert "actual_pelvis_pos[:2]" in root_dev_source
    assert "current_ref_pelvis_pos[:2]" in root_dev_source


def test_stair_contains_z_agnostic_elastic_pacing_contract():
    text = source("stair")
    assert "class NPZTrajectoryAdapter" in text
    assert "_elastic_reference_step" in text
    assert "planar_root_deviation < 0.4" in text
    assert "self._slow_wait_counter % 2 == 0" in text
    assert "data.qpos[:2] - ref_data.qpos[:2]" in text
    assert "_residual_l2_weight = 0.02" in text
    assert "metabolic_weight" not in text
    assert "metabolic_smoothing" not in text
    assert "from .common import" not in text


def test_run_paper_reward_helpers():
    relax = load_pure_function("run", "_run_tracking_distance")
    propel = load_pure_function("run", "_run_propulsive_reward")
    impact = load_pure_function("run", "_run_impact_penalty")

    assert relax(2.0, False) == 2.0
    assert relax(2.0, True) == 1.0
    assert propel(4.0, 3.0, True) == 0.3
    assert propel(2.0, 3.0, True) == 0.0
    assert propel(4.0, 3.0, False) == 0.0
    assert impact(2.5) == 0.0
    assert impact(3.5) == -0.1


def test_run_contains_standalone_safety_and_flight_contract():
    text = source("run")
    assert "class NPZTrajectoryAdapter" in text
    assert "def _apply_qacc_shield(" in text
    assert "flight_phase = bool(" in text
    assert "vertical_grf_body_weights" in text
    assert "propulsive_velocity_weight: float = 0.3" in text
    assert "grf_penalty_weight: float = 0.1" in text
    assert "grf_limit_body_weights: float = 2.5" in text
    assert "_residual_l2_weight = 0.02" in text
    assert "metabolic_weight" not in text
    assert "metabolic_smoothing" not in text
    assert "from .common import" not in text
    assert (
        'else "rl_paradigms.residualrl.run"'
        in text
    )


def _minimal_run_contact_model(foot_height: float) -> mujoco.MjModel:
    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body name="foot" pos="0 0 {foot_height}">
          <freejoint/>
          <geom name="foot_geom" type="sphere" size="0.1" mass="2"/>
        </body>
        <body name="other" pos="1 0 1">
          <geom name="other_geom" type="sphere" size="0.05" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _run_contact_wrapper(model: mujoco.MjModel, data: mujoco.MjData):
    wrapper_class = load_ast_class("run", "MSKBenchResidualRunWrapper")
    wrapper = object.__new__(wrapper_class)
    foot_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "foot",
    )
    other_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "other",
    )
    wrapper.env = SimpleNamespace(
        unwrapped=SimpleNamespace(_model=model, _data=data)
    )
    wrapper._foot_body_ids = frozenset((foot_id,))
    wrapper._left_foot_body_ids = (foot_id,)
    wrapper._right_foot_body_ids = ()
    wrapper._support_geom_ids = wrapper._infer_support_geom_ids(model)
    return wrapper, foot_id, other_id


def test_run_contact_classifier_and_grf_share_ground_support_logic(monkeypatch):
    model = _minimal_run_contact_model(foot_height=0.09)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0

    wrapper, foot_id, other_id = _run_contact_wrapper(model, data)
    foot_geom = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "foot_geom",
    )
    other_geom = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "other_geom",
    )
    ground_geom = next(iter(wrapper._support_geom_ids))

    assert wrapper._is_foot_support_contact(foot_geom, ground_geom)
    assert wrapper._is_foot_support_contact(ground_geom, foot_geom)
    assert not wrapper._is_foot_support_contact(other_geom, ground_geom)
    assert wrapper._foot_contact((foot_id,))
    assert not wrapper._foot_contact((other_id,))
    flight_phase = bool(
        not wrapper._left_foot_contact()
        and not wrapper._right_foot_contact()
    )
    assert not flight_phase

    vertical_force = 30.0

    def fake_contact_force(_model, contact_data, contact_index, out):
        contact_frame = np.asarray(
            contact_data.contact[contact_index].frame,
            dtype=np.float64,
        ).reshape(3, 3)
        out[:] = 0.0
        out[:3] = contact_frame @ np.array(
            [0.0, 0.0, vertical_force],
            dtype=np.float64,
        )

    monkeypatch.setattr(mujoco, "mj_contactForce", fake_contact_force)
    body_weight = float(
        np.sum(model.body_mass) * abs(model.opt.gravity[2])
    )
    assert wrapper._vertical_grf_body_weights() == pytest.approx(
        vertical_force / body_weight
    )

    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    assert not wrapper._foot_contact((foot_id,))
    assert bool(
        not wrapper._left_foot_contact()
        and not wrapper._right_foot_contact()
    )


def test_run_rejects_models_without_explicit_support_geometries():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="foot" pos="0 0 1">
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    wrapper_class = load_ast_class("run", "MSKBenchResidualRunWrapper")
    with pytest.raises(ValueError, match="support"):
        wrapper_class._infer_support_geom_ids(model)
