from typing import Any, Union, Tuple, Dict
from types import ModuleType

import numpy as np
import jax
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.control_functions import ControlFunction
from flax import struct


class DefaultControl(ControlFunction):

    """
    Uses the default actuator from the environment. This controller internally normalizes the action space to [-1, 1]
    for the agent but uses the original action space for the environment.
    """

    @struct.dataclass
    class DefaultControlState:
        prev_ctrl: Union[np.ndarray, jax.Array]

    def __init__(self, env: any, apply_mode: str = "direct", **kwargs: Dict):
        """
        Initialize the control function class.

        Args:
            env (Any): The environment instance.
            apply_mode (str): How to apply control. "direct" rescales to actuator space (default).
                              "incremental" integrates deltas over previous control with clipping.
            **kwargs (Dict): Additional keyword arguments.
        """
        # get the limits of the action space
        self._actuator_low, self._actuator_high = self._get_actuator_limits(env._action_indices, env._model)

        # calculate mean and delta
        self.norm_act_mean = (self._actuator_high + self._actuator_low) / 2.0
        self.norm_act_delta = (self._actuator_high - self._actuator_low) / 2.0

        # application mode
        if apply_mode not in ("direct", "incremental"):
            raise ValueError(f"Invalid apply_mode '{apply_mode}'. Use 'direct' or 'incremental'.")
        self._apply_mode = apply_mode

        # set the action space limits for the agent to -1 and 1
        low = -np.ones_like(self.norm_act_mean)
        high = np.ones_like(self.norm_act_mean)

        super(DefaultControl, self).__init__(env, low, high, **kwargs)

    def init_state(self,
                   env: Any,
                   key: Union[jax.random.PRNGKey, Any],
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> "DefaultControlState":
        """
        Initialize controller state. For incremental mode, keep previous control in actuator space.
        """
        # Start from the mid-point of actuator range by default
        return DefaultControl.DefaultControlState(prev_ctrl=backend.asarray(self.norm_act_mean))

    def generate_action(self, env: Any,
                        action: Union[np.ndarray, jax.Array],
                        model: Union[MjModel, Model],
                        data: Union[MjData, Data],
                        carry: Any,
                        backend: ModuleType) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """
        Calculates the action. This function scales the action from [-1, 1] to the original action space.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jax.Array]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The action and carry.

        """
        if self._apply_mode == "direct":
            # Standard behavior: rescale from [-1, 1] to actuator space
            unnormalized_action = self._unnormalize_action(action)
            # Safety-clip to actuator limits
            ctrl = backend.clip(unnormalized_action, backend.asarray(self._actuator_low), backend.asarray(self._actuator_high))
            return ctrl, carry

        # Incremental mode: a_t = clip_{[low,high]}(m_{t-1} + clip_{[-1,1]}(alpha_t') * norm_act_delta)
        # where m_{t-1} is previous control in actuator space
        state = carry.control_func_state
        alpha_clipped = backend.clip(action, -1.0, 1.0)
        delta = alpha_clipped * backend.asarray(self.norm_act_delta)
        proposed = state.prev_ctrl + delta
        ctrl = backend.clip(proposed, backend.asarray(self._actuator_low), backend.asarray(self._actuator_high))
        # update state
        new_state = DefaultControl.DefaultControlState(prev_ctrl=ctrl)
        carry = carry.replace(control_func_state=new_state)
        return ctrl, carry

    def _unnormalize_action(self, action: Union[np.ndarray, jax.Array]) -> Union[np.ndarray, jax.Array]:
        """
        Rescale the action from [-1, 1] to the desired action space.

        Args:
            action (Union[np.ndarray, jax.Array]): The action to be unnormalized.

        Returns:
            Union[np.ndarray, jax.Array]: The unnormalized action

        """
        unnormalized_action = ((action * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action
