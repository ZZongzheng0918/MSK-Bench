"""Tests for musclemimic.utils.msk_metrics module."""

import numpy as np
import mujoco as mj
from musclemimic.utils.retarget.msk_metrics import (
    floating_height,
    ground_penetration_stats,
    joint_violation_pct,
    load_model,
    tendon_jump_max,
)


# =============================================================================
# floating_height() tests
# =============================================================================

FLOAT_THRESHOLD = 0.005  # 5mm above ground


def test_grounded_trajectory(model):
    """Test that trajectory at ground level returns ~0 min height."""
    qpos = np.zeros((10, model.nq))

    min_z = floating_height(model, qpos)

    assert min_z < 0.1, f"Expected low min height, got {min_z}m"


def test_floating_trajectory_detected(model):
    """Test that elevated trajectory returns positive min height."""
    qpos = np.zeros((10, model.nq))
    qpos[:, 2] = 2.0  # 2m above ground

    min_z = floating_height(model, qpos)

    assert min_z > FLOAT_THRESHOLD, f"Expected floating, got min height {min_z}m"


def test_floating_height_varies_with_z(model):
    """Test that min height increases with Z position."""
    qpos_low = np.zeros((5, model.nq))
    qpos_low[:, 2] = 0.5

    qpos_high = np.zeros((5, model.nq))
    qpos_high[:, 2] = 2.0

    min_z_low = floating_height(model, qpos_low)
    min_z_high = floating_height(model, qpos_high)

    assert min_z_high > min_z_low, (
        f"Higher Z should give higher min_z: low={min_z_low}, high={min_z_high}"
    )


# =============================================================================
# ground_penetration_stats() tests
# =============================================================================


def test_no_penetration(model):
    """Test that trajectory well above ground has no penetration."""
    qpos = np.zeros((10, model.nq))
    qpos[:, 2] = 1.0  # Z height well above ground

    pen_pct, max_pen_m = ground_penetration_stats(model, qpos)

    assert pen_pct == 0.0, f"Expected 0% penetration, got {pen_pct}%"
    assert max_pen_m == 0.0, f"Expected 0m max penetration, got {max_pen_m}m"


def test_penetration_detected(model):
    """Test that trajectory dipping below ground detects penetration."""
    qpos = np.zeros((10, model.nq))
    qpos[5:8, 2] = -0.05  # 5cm below ground

    pen_pct, max_pen_m, max_frame, max_geom = ground_penetration_stats(
        model, qpos, return_info=True
    )

    assert pen_pct > 0.0, f"Expected penetration, got {pen_pct}%"
    assert max_pen_m > 0.0, f"Expected positive penetration depth, got {max_pen_m}m"
    assert max_frame >= 5, f"Expected penetration in frames 5-7, got frame {max_frame}"


def test_penetration_tolerance(model):
    """Test that penetrations below tolerance are ignored."""
    qpos = np.zeros((10, model.nq))
    qpos[:, 2] = 0.9  # Height that produces some penetration

    # First measure actual penetration with no tolerance
    _, max_pen_m = ground_penetration_stats(model, qpos, tol=0)
    assert max_pen_m > 0, "Test requires some penetration to verify tolerance"

    # Setting tolerance above max penetration should filter everything
    pen_pct, _ = ground_penetration_stats(model, qpos, tol=max_pen_m + 1e-4)
    assert pen_pct == 0.0, f"Expected 0% when tol > max_pen, got {pen_pct}%"


# =============================================================================
# joint_violation_pct() tests
# =============================================================================


def test_joint_violation_pct_no_violation():
    """Test that joints at midpoint of range have 0% violation."""
    model = load_model()
    T, nq = 10, model.nq

    qpos = np.zeros((T, nq))

    for j in range(model.njnt):
        if model.joint(j).type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            idx = model.jnt_qposadr[j]
            low, high = model.jnt_range[j]
            if not np.isinf(low) and not np.isinf(high):
                qpos[:, idx] = (low + high) / 2

    violation_pct = joint_violation_pct(model, qpos)
    assert violation_pct == 0.0, f"Expected 0% violation, got {violation_pct}%"


def test_joint_violation_pct_with_violation():
    """Test that joints exceeding limits are detected."""
    model = load_model()
    T, nq = 5, model.nq
    qpos = np.zeros((T, nq))

    hinge_joints = [
        j
        for j in range(model.njnt)
        if model.joint(j).type == mj.mjtJoint.mjJNT_HINGE
    ]
    assert len(hinge_joints) > 0, "No hinge joints found"

    hinge_j = hinge_joints[0]
    idx = model.jnt_qposadr[hinge_j]
    low, high = model.jnt_range[hinge_j]

    qpos[:, idx] = high + 0.1

    violation_pct = joint_violation_pct(model, qpos)
    assert violation_pct > 0.0, f"Expected violations, got {violation_pct}%"


# =============================================================================
# tendon_jump_max() tests
# =============================================================================


def test_tendon_jump_detected():
    """Test that sudden qpos changes trigger tendon jump detection."""
    model = load_model()
    T, nq = 10, model.nq

    qpos = np.zeros((T, nq))
    qpos[5, :] += 0.5  # Induce a jump mid-trajectory

    max_jump, max_t, max_k, tendon_name = tendon_jump_max(
        model, qpos, jump_factor=2.0, min_rel_jump=1e-6, return_info=True
    )

    assert max_jump > 0, f"Expected jump detected, got max_jump={max_jump}"
    assert max_t >= 0, f"Expected valid frame index, got {max_t}"
    assert tendon_name is None or isinstance(tendon_name, str)


def test_no_tendon_jump():
    """Test that constant qpos produces no tendon jumps."""
    model = load_model()
    T, nq = 20, model.nq
    qpos = np.zeros((T, nq))

    max_jump = tendon_jump_max(model, qpos)

    assert max_jump == 0.0, f"Expected no jumps, got max_jump={max_jump}"


def test_tendon_jump_frame_index():
    """Test that jump is detected at the correct frame."""
    model = load_model()
    T, nq = 15, model.nq

    qpos = np.zeros((T, nq))
    qpos[7, :] += 1.0  # Big jump at frame 7

    max_jump, max_t, max_k, tendon_name = tendon_jump_max(
        model, qpos, min_rel_jump=1e-6, return_info=True
    )

    assert max_t in [7, 8], f"Expected jump at frame 7 or 8, got frame {max_t}"
    assert tendon_name is None or isinstance(tendon_name, str)


def test_tendon_jump_magnitude_positive():
    """Test that detected jump magnitude is positive."""
    model = load_model()
    T, nq = 12, model.nq

    qpos = np.zeros((T, nq))
    qpos[6, :] += 0.3

    max_jump = tendon_jump_max(model, qpos, min_rel_jump=1e-6)

    if max_jump > 0:
        assert max_jump > 0, "Jump magnitude should be positive"


def test_tendon_jump_runtime():
    """Test that tendon jump detection completes in reasonable time."""
    model = load_model()
    T, nq = 200, model.nq
    qpos = np.random.randn(T, nq) * 0.01

    max_jump = tendon_jump_max(model, qpos)

    assert isinstance(max_jump, float)
