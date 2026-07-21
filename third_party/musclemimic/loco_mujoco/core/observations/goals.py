from copy import deepcopy
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from jax.scipy.spatial.transform import Rotation as jnp_R
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.observations.base import Observation, StatefulObservation
from loco_mujoco.core.observations.visualizer import RootVelocityArrowVisualizer
from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.utils.math import (
    calculate_relative_site_quantities,
    quat_scalarfirst2scalarlast,
)
from loco_mujoco.core.utils.mujoco import mj_jntid2qposid, mj_jntid2qvelid, mj_jntname2qposid, mj_jntname2qvelid


class Goal(StatefulObservation):
    """
    Base class representing a goal in the environment.

    Args:
        info_props (Dict): Information properties required for initialization.
        visualize_goal (bool): Whether to visualize the goal.
        n_visual_geoms (int): Number of visual geometries for visualization.
    """

    def __init__(self, info_props: dict, visualize_goal: bool = False, n_visual_geoms: int = 0, **kwargs):
        self._initialized_from_traj = False
        self._info_props = info_props
        if visualize_goal:
            assert self.has_visual, (
                f"{self.__class__.__name__} does not support visualization. Please set visualize_goal to False."
            )
        self.visualize_goal = visualize_goal

        # Filter out parameters that are not accepted by the base Observation class
        observation_kwargs = {k: v for k, v in kwargs.items() if k in ["group", "allow_randomization"]}
        if "group" not in observation_kwargs:
            observation_kwargs["group"] = "goal"
        Observation.__init__(self, obs_name=self.__class__.__name__, **observation_kwargs)
        StatefulObject.__init__(self, n_visual_geoms)

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization. Needs to be implemented in subclasses."""
        raise NotImplementedError

    @property
    def requires_trajectory(self) -> bool:
        """Check if the goal requires a trajectory."""
        return False

    @classmethod
    def data_type(cls) -> Any:
        """Return the data type used by this goal."""
        return None

    def reset_state(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[MjData | Any, Any]:
        """
        Reset the state of the goal.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[MjData, Any], Any]: Updated data and carry.
        """
        assert self.initialized
        return data, carry

    def is_done(self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType) -> bool:
        """
        Check if the goal is completed.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            bool: Whether the goal is done.
        """
        return False

    def mjx_is_done(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> bool:
        """
        Check if the goal is done (jax-compatible version).

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            bool: Whether the goal is done.
        """
        return False

    def apply_spec_modifications(self, spec: MjSpec, info_props: dict) -> MjSpec:
        """
        Apply modifications to the Mujoco XML specification to include the goal.

        Args:
            spec (MjSpec): Mujoco specification.
            info_props (Dict): Information properties.

        Returns:
            MjSpec: Modified Mujoco specification.
        """
        return spec

    def set_attr_compat(
        self,
        data: MjData | Data,
        backend: ModuleType,
        attr: str,
        arr: np.ndarray | jnp.ndarray,
        ind: np.ndarray | jnp.ndarray | None = None,
    ) -> MjData | Any:
        """
        Set attributes in a backend-compatible manner.

        Args:
            data (Union[MjData, Data]): Data object to modify.
            backend (ModuleType): Backend to use (numpy or jax).
            attr (str): Attribute name to modify.
            arr (Union[np.ndarray, jnp.ndarray]): Array to set.
            ind (Union[np.ndarray, jnp.ndarray, None]): Indices to modify.

        Returns:
            Union[MjData, Any]: Modified data.
        """
        if ind is None:
            ind = backend.arange(len(arr))

        if backend == np:
            getattr(data, attr)[ind] = arr
        elif backend == jnp:
            data = data.replace(**{attr: getattr(data, attr).at[ind].set(arr)})
        else:
            raise NotImplementedError
        return data

    @property
    def initialized(self) -> bool:
        """Check if the goal is initialized."""
        init_from_traj = True if not self.requires_trajectory else self._initialized_from_traj
        return self._initialized_from_mj and init_from_traj

    @property
    def dim(self) -> int:
        """Get the dimension of the goal."""
        raise NotImplementedError

    @property
    def requires_spec_modification(self) -> bool:
        """Check if the goal requires specification modification."""
        return self.__class__.apply_spec_modifications != Goal.apply_spec_modifications

    @classmethod
    def list_goals(cls) -> list:
        """List all subclasses of Goal."""
        return [goal for goal in Goal.__subclasses__()]


class NoGoal(Goal):
    """
    Empty goal class.
    """

    def _init_from_mj(self, env: Any, model: MjModel | Any, data: MjData | Any, current_obs_size: int):
        """
        Initialize the class from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.data_type_ind = np.array([])
        self.obs_ind = np.array([])
        self._initialized_from_mj = True

    def get_obs_and_update_state(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[np.ndarray | jnp.ndarray, Any]:
        """
        Get the observation and update the state. Always returns an empty array for NoGoal.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Empty observation and unchanged carry.
        """
        return backend.array([]), carry

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization. Always False for NoGoal."""
        return False

    @property
    def dim(self) -> int:
        """Get the dimension of the goal. Always 0 for NoGoal."""
        return 0


@struct.dataclass
class GoalRandomRootVelocityState:
    """
    State class for random root velocity goal.

    Attributes:
        goal_vel_x (float): Goal velocity in the x direction.
        goal_vel_y (float): Goal velocity in the y direction.
        goal_vel_yaw (float): Goal yaw velocity.
    """

    goal_vel_x: float
    goal_vel_y: float
    goal_vel_yaw: float


class GoalRandomRootVelocity(Goal, RootVelocityArrowVisualizer):
    """
    A class representing a random root velocity goal.

    This class defines a goal that specifies random velocities for the root body in
    the x, y, and yaw directions.

    Args:
        info_props (Dict): Information properties required for initialization.
        max_x_vel (float): Maximum velocity in the x direction.
        max_y_vel (float): Maximum velocity in the y direction.
        max_yaw_vel (float): Maximum yaw velocity.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self, info_props: dict, max_x_vel: float = 1.0, max_y_vel: float = 1.0, max_yaw_vel: float = 1.0, **kwargs
    ):
        self._traj_goal_ind = None
        self.max_x_vel = max_x_vel
        self.max_y_vel = max_y_vel
        self.max_yaw_vel = max_yaw_vel
        self.upper_body_xml_name = info_props["upper_body_xml_name"]
        self.free_jnt_name = info_props["root_free_joint_xml_name"]

        # To be initialized from Mujoco
        self._root_body_id = None
        self._root_jnt_qpos_start_id = None

        # call visualizer init
        RootVelocityArrowVisualizer.__init__(self, info_props)

        # call goal init
        n_visual_geoms = (
            self._arrow_n_visual_geoms if "visualize_goal" in kwargs.keys() and kwargs["visualize_goal"] else 0
        )
        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

    def _init_from_mj(self, env: Any, model: MjModel | Model, data: MjData | Data, current_obs_size: int):
        """
        Initialize the goal from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.upper_body_xml_name)
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(self.free_jnt_name, model))
        self._initialized_from_mj = True

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization."""
        return True

    def init_state(
        self, env: Any, key: jax.random.PRNGKey, model: MjModel | Model, data: MjData | Data, backend: ModuleType
    ) -> GoalRandomRootVelocityState:
        """
        Initialize the goal state.

        Args:
            env (Any): The environment instance.
            key (jax.random.PRNGKey): Random key for sampling.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            GoalRandomRootVelocityState: Initialized state.
        """
        return GoalRandomRootVelocityState(0.0, 0.0, 0.0)

    def reset_state(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[MjData | Any, Any]:
        """
        Reset the goal state with random velocities.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[MjData, Any], Any]: Updated data and carry.
        """
        key = carry.key
        if backend == np:
            goal_vel = np.random.uniform(
                [-self.max_x_vel, -self.max_y_vel, -self.max_yaw_vel],
                [self.max_x_vel, self.max_y_vel, self.max_yaw_vel],
            )
        else:
            key, subkey = jax.random.split(key)
            goal_vel = jax.random.uniform(
                subkey,
                shape=(3,),
                minval=jnp.array([-self.max_x_vel, -self.max_y_vel, -self.max_yaw_vel]),
                maxval=jnp.array([self.max_x_vel, self.max_y_vel, self.max_yaw_vel]),
            )

        goal_state = GoalRandomRootVelocityState(goal_vel[0], goal_vel[1], goal_vel[2])
        observation_states = carry.observation_states.replace(**{self.name: goal_state})
        return data, carry.replace(key=key, observation_states=observation_states)

    def get_obs_and_update_state(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[np.ndarray | jnp.ndarray, Any]:
        """
        Get the current goal observation and update the state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Goal observation and updated carry.
        """
        goal_vel_x = getattr(carry.observation_states, self.name).goal_vel_x
        goal_vel_y = getattr(carry.observation_states, self.name).goal_vel_y
        goal_vel_yaw = getattr(carry.observation_states, self.name).goal_vel_yaw
        goal = backend.array([goal_vel_x, goal_vel_y, goal_vel_yaw])
        goal_visual = backend.array([goal_vel_x, goal_vel_y, 0.0, 0.0, 0.0, goal_vel_yaw])

        if self.visualize_goal:
            carry = self.set_visuals(
                goal_visual,
                env,
                model,
                data,
                carry,
                self._root_body_id,
                self._free_jnt_qpos_id,
                self.visual_geoms_idx,
                backend,
            )

        return goal, carry

    @property
    def dim(self) -> int:
        """Get the dimension of the goal."""
        return 3


@struct.dataclass
class GoalTrajRootVelocityState:
    """
    State class for trajectory root velocity goal.

    Attributes:
        goal_vel (Union[np.ndarray, jnp.ndarray]): Velocity goal for the root.
    """

    goal_vel: np.ndarray | jnp.ndarray


class GoalTrajRootVelocity(Goal, RootVelocityArrowVisualizer):
    """
    A class representing a trajectory-based root velocity goal.

    This class defines a goal that computes the root velocity based on trajectory data
    and averages over a specified number of future steps.

    Args:
        info_props (Dict): Information properties required for initialization.
        n_steps_average (int): Number of future steps to average over for the velocity goal.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, info_props: dict, n_steps_average: int = 3, **kwargs):
        self._traj_goal_ind = None
        self.upper_body_xml_name = info_props["upper_body_xml_name"]
        self.free_jnt_name = info_props["root_free_joint_xml_name"]

        # To be initialized from Mujoco
        self._root_body_id = None
        self._root_jnt_qpos_start_id = None
        self._free_jnt_qvelid = None
        self._free_jnt_qposid = None

        # Number of future steps in the trajectory to average the goal over
        self._n_steps_average = n_steps_average

        # Call visualizer initialization (if applicable)
        RootVelocityArrowVisualizer.__init__(self, info_props)

        # Call goal initialization
        n_visual_geoms = (
            self._arrow_n_visual_geoms if "visualize_goal" in kwargs.keys() and kwargs["visualize_goal"] else 0
        )

        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

    def _init_from_mj(self, env: Any, model: MjModel | Model, data: MjData | Data, current_obs_size: int):
        """
        Initialize the goal from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True
        self._root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.upper_body_xml_name)
        self._free_jnt_qposid = np.array(mj_jntname2qposid(self.free_jnt_name, model))
        self._free_jnt_qvelid = np.array(mj_jntname2qvelid(self.free_jnt_name, model))

    def init_from_traj(self, traj_handler: Any):
        """
        Initialize from a trajectory handler.

        Args:
            traj_handler (Any): The trajectory handler.
        """
        assert traj_handler is not None, (
            f"Trajectory handler is None, using {__class__.__name__} requires a trajectory."
        )
        self._initialized_from_traj = True

    def init_state(
        self, env: Any, key: jax.random.PRNGKey, model: MjModel | Any, data: MjData | Any, backend: ModuleType
    ) -> GoalTrajRootVelocityState:
        """
        Initialize the goal state.

        Args:
            env (Any): The environment instance.
            key (jax.random.PRNGKey): Random key for sampling.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            GoalTrajRootVelocityState: Initialized state with zero velocity.
        """
        return GoalTrajRootVelocityState(backend.zeros(self.dim))

    def get_obs_and_update_state(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[np.ndarray | jnp.ndarray, Any]:
        """
        Get the current goal observation and update the state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Goal observation and updated carry.
        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # Get trajectory data and state
        traj_data = env.th.traj.data
        traj_state = carry.traj_state

        # Get a slice of the trajectory data
        traj_qpos = backend.atleast_2d(
            traj_data.get_qpos_slice(traj_state.traj_no, traj_state.subtraj_step_no, self._n_steps_average, backend)
        )
        traj_qvel = backend.atleast_2d(
            traj_data.get_qvel_slice(traj_state.traj_no, traj_state.subtraj_step_no, self._n_steps_average, backend)
        )

        # Get the average goal over the slice
        traj_free_jnt_qpos = traj_qpos[0, self._free_jnt_qposid]
        traj_free_jnt_qvel = traj_qvel[:, self._free_jnt_qvelid]
        traj_free_jnt_lin_vel = backend.mean(traj_free_jnt_qvel[:, :3], axis=0)
        traj_free_jnt_rot_vel = backend.mean(traj_free_jnt_qvel[:, 3:], axis=0)
        traj_free_jnt_quat = traj_free_jnt_qpos[3:]
        traj_free_jnt_mat = R.from_quat(quat_scalarfirst2scalarlast(traj_free_jnt_quat)).as_matrix()

        # Transform lin and rot vel to local frame
        traj_free_jnt_lin_vel = traj_free_jnt_mat.T @ traj_free_jnt_lin_vel
        traj_free_jnt_rot_vel = traj_free_jnt_mat.T @ traj_free_jnt_rot_vel

        goal = backend.concatenate([traj_free_jnt_lin_vel, traj_free_jnt_rot_vel])

        if self.visualize_goal:
            carry = self.set_visuals(
                goal, env, model, data, carry, self._root_body_id, self._free_jnt_qposid, self.visual_geoms_idx, backend
            )

        return goal, carry

    def is_done(self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType) -> bool:
        """
        Check if the goal is completed.

        Terminates the episode if the number of steps till the end of the trajectory is
        less than the number of steps to average over.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            bool: Whether the goal is done.
        """
        steps_till_end = self._steps_till_end(env.th.traj.data, carry.traj_state)
        return steps_till_end < self._n_steps_average

    def mjx_is_done(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> bool:
        """
        Check if the goal is done (JAX-compatible).

        Terminates the episode if the number of steps till the end of the trajectory is
        less than the number of steps to average over.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            bool: Whether the goal is done.
        """
        steps_till_end = self._steps_till_end(env.th.traj.data, carry.traj_state)
        return jax.lax.cond(steps_till_end < self._n_steps_average, lambda: True, lambda: False)

    def _steps_till_end(self, traj_data: Any, traj_state: Any) -> int:
        """
        Calculate the number of steps till the end of the trajectory.

        Args:
            traj_data (Any): Trajectory data.
            traj_state (Any): Current trajectory state.

        Returns:
            int: Number of steps till the end of the trajectory.
        """
        traj_no = traj_state.traj_no
        subtraj_step_no = traj_state.subtraj_step_no
        current_idx = traj_data.split_points[traj_no] + subtraj_step_no
        idx_of_next_traj = traj_data.split_points[traj_no + 1]
        return idx_of_next_traj - current_idx

    @classmethod
    def get_all_obs_of_type(cls, model: MjModel | Model, data: MjData | Data, ind: Any, backend: ModuleType) -> Any:
        """
        Retrieve all observations of this type.

        Args:
            model (Union[MjModel, Any]): The Mujoco model.
            data (Union[MjData, Any]): The Mujoco data.
            ind (Any): Index for observations.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Any: Flattened observations.
        """
        return backend.ravel(data.userdata[ind.GoalTrajRootVelocity])

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization."""
        return True

    @property
    def requires_trajectory(self) -> bool:
        """Check if the goal requires a trajectory."""
        return True

    def set_visual_data(
        self, data: MjData | Data, backend: ModuleType, traj_goal: np.ndarray | jnp.ndarray
    ) -> MjData | Any:
        """
        Set visualization data for the goal.

        Args:
            data (Union[MjData, Data]): The Mujoco data.
            backend (ModuleType): The backend (numpy or jax).
            traj_goal (Union[np.ndarray, jnp.ndarray]): The trajectory goal.

        Returns:
            Union[MjData, Any]: Updated Mujoco data with visualization settings.
        """
        rel_target_arrow_pos = backend.concatenate([traj_goal * self._arrow_to_goal_ratio, jnp.ones(1)])
        abs_target_arrow_pos = (
            data.body(self.upper_body_xml_name).xmat.reshape(3, 3) @ rel_target_arrow_pos
        ) + data.body(self.upper_body_xml_name).xpos
        data.site(self._site_name_keypoint_2).xpos = abs_target_arrow_pos
        return data

    @property
    def dim(self) -> int:
        """Get the dimension of the goal."""
        return 6


__all__ = [
    "Goal",
    "GoalRandomRootVelocity",
    "GoalTrajRootVelocity",
    "NoGoal",
]
