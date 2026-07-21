from typing import Any, Union, Tuple
from types import ModuleType

import numpy as np
import jax.numpy as jnp
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.terrain import Terrain
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class StaticTerrain(Terrain):
    """
    Static terrain class inheriting from Terrain. This class is used for terrains that do not change over time
    (e.g., flat terrain).

    """

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.
        """
        assert_backend_is_supported(backend)
        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.
        """
        assert_backend_is_supported(backend)
        return model, data, carry

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """
        Modify the simulation specification.

        Args:
            spec (MjSpec): The simulation specification.

        Returns:
            MjSpec: The unmodified simulation specification.
        """
        return spec

    @property
    def is_dynamic(self) -> bool:
        """
        Check if the terrain is dynamic.

        Returns:
            bool: False, as this terrain is static.
        """
        return False

    def sample_heights_at_points(self,
                                  x: Union[np.ndarray, jnp.ndarray],
                                  y: Union[np.ndarray, jnp.ndarray],
                                  model: Union[MjModel, Model],
                                  carry: Any,
                                  backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        """
        Sample terrain heights at given world (x, y) coordinates.

        For static (flat) terrain, returns zero height everywhere.

        Args:
            x: Array of x coordinates in world frame.
            y: Array of y coordinates in world frame.
            model: The simulation model (unused).
            carry: Carry instance (unused - no terrain_state for static terrain).
            backend: Backend module (numpy or jax.numpy).

        Returns:
            Array of zeros with same shape as x.
        """
        # Static terrain is flat at z=0, no need to access carry.terrain_state
        return backend.zeros_like(x)
