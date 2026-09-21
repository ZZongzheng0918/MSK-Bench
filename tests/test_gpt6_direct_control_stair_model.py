from __future__ import annotations

import mujoco

from rl_paradigms.residualrl.stair import MSKBenchResidualStairEnvV0


def test_residual_stair_environment_uses_bundled_two_step_model():
    env = MSKBenchResidualStairEnvV0()
    try:
        geom_names = {
            env._model.geom(index).name
            for index in range(env._model.ngeom)
        }
        assert {"stair_step_1", "stair_step_2"} <= geom_names
        assert mujoco.mj_name2id(
            env._model,
            mujoco.mjtObj.mjOBJ_BODY,
            "two_step_staircase",
        ) >= 0
    finally:
        env.close()
