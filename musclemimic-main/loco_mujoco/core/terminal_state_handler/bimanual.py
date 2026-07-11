from typing import Dict, Any, Union, Tuple
from types import ModuleType

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class BimanualTerminalStateHandler(TerminalStateHandler):
    """
    Terminal state handler for bimanual manipulation models.
    
    Since bimanual models are fixed in place for manipulation tasks and don't move around 
    like full humanoids, this handler skips traditional locomotion-based terminal state 
    checks such as height deviations, position wandering, or falling detection that are 
    appropriate for walking humanoids.
    
    This handler essentially provides no terminal states for bimanual models, allowing
    them to focus on manipulation tasks without premature episode termination.
    """

    registered: Dict[str, type] = dict()

    def __init__(self, env: Any, **handler_config: Dict[str, Any]):
        """
        Initialize the BimanualTerminalStateHandler.

        Args:
            env (Any): The environment instance.
            **handler_config (Any): Configuration dictionary.
        """
        self._info_props = env._get_all_info_properties()
        self._handler_config = handler_config

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the terminal state handler.

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

    def is_absorbing(self,
                     env: Any,
                     obs: np.ndarray,
                     info: Dict[str, Any],
                     data: MjData,
                     carry: Any) -> Union[bool, Any]:
        """
        Always returns false for bimanual models. Function for CPU Mujoco.

        Args:
            env (Any): The environment instance.
            obs (np.ndarray): Observations with shape (n_samples, n_obs).
            info (Dict[str, Any]): The info dictionary.
            data (MjData): The Mujoco data structure.
            carry (Any): Additional carry information.

        Returns:
            Union[bool, Any]: Whether the current state is terminal (always False), and the carry.

        """
        return False, carry

    def mjx_is_absorbing(self,
                         env: Any,
                         obs: jnp.ndarray,
                         info: Dict[str, Any],
                         data: Data,
                         carry: Any) -> Union[bool, Any]:
        """
        Always returns false for bimanual models. Function for Mjx.

        Args:
            env (Any): The environment instance.
            obs (jnp.ndarray): Observations with shape (n_samples, n_obs).
            info (Dict[str, Any]): The info dictionary.
            data (Data): The Mujoco data structure for Mjx.
            carry (Any): Additional carry information.

        Returns:
            Union[bool, Any]: Whether the current state is terminal (always False), and the carry.

        """
        return False, carry
