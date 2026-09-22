from types import ModuleType
from typing import Any, Union, Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model
import scipy as np_scipy
import jax.scipy as jnp_scipy

from loco_mujoco.core.terrain import DynamicTerrain
from loco_mujoco.core.utils import mj_jntname2qposid
from loco_mujoco.core.utils.backend import assert_backend_is_supported


@struct.dataclass
class RoughTerrainState:
    """
    Represents the state of the rough terrain.

    Attributes:
        height_field_raw (Union[np.ndarray, jax.Array]): The raw height field data.
    """
    height_field_raw: Union[np.ndarray, jax.Array]


class RoughTerrain(DynamicTerrain):
    """
    Dynamic rough terrain class for simulating uneven surfaces. This terrain is generated randomly on
    each reset.

    """

    viewer_needs_to_update_hfield: bool = True

    def __init__(self, env: Any,
                 inner_platform_size_in_meters: float = 1.0,
                 random_min_height: float = -0.05,
                 random_max_height: float = 0.05,
                 random_step: float = 0.005,
                 random_downsampled_scale: float = 0.4,
                 hfield_length: int = 80,
                 **kwargs: Any):
        """
        Initialize the rough terrain.

        Args:
            env (Any): The environment instance.
            inner_platform_size_in_meters (float): Size of the inner platform in meters.
            random_min_height (float): Minimum random height for terrain generation.
            random_max_height (float): Maximum random height for terrain generation.
            random_step (float): Step size for random height values.
            random_downsampled_scale (float): Downsample scale for terrain.
            hfield_length (int): Heightfield grid resolution. Lower values = coarser terrain = fewer contacts.
                                 Default 80 gives 10cm cells over 8m. Use 40 for 20cm cells to reduce contacts.
            **kwargs (Any): Additional arguments for initialization.
        """
        super().__init__(env, **kwargs)

        self.inner_platform_size_in_meters = inner_platform_size_in_meters
        self.random_min_height = random_min_height
        self.random_max_height = random_max_height
        self.random_step = random_step
        self.random_downsampled_scale = random_downsampled_scale

        self.hfield_size = (4, 4, 30.0, 0.125)
        self.hfield_length = hfield_length
        self.hfield_half_length_in_meters = self.hfield_size[0]  # radius_x = radius_y = 4m
        self.elevation_z = self.hfield_size[2]  # 30.0 - maximum height scale

        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.elevation_z

        # Geom position (z-coordinate where hfield geom is placed).
        # Used in both modify_spec() and sample_heights_at_points().
        self.geom_z = -0.06

        self.heights_range = jnp.arange(self.random_min_height, self.random_max_height + self.random_step,
                                        self.random_step)

        self.x = jnp.linspace(0, self.hfield_length, int(self.hfield_length * self.random_downsampled_scale))
        self.y = jnp.linspace(0, self.hfield_length, int(self.hfield_length * self.random_downsampled_scale))
        x_upsampled = jnp.linspace(0, self.hfield_length, self.hfield_length)
        y_upsampled = jnp.linspace(0, self.hfield_length, self.hfield_length)
        x_upsampled_grid, y_upsampled_grid = jnp.meshgrid(x_upsampled, y_upsampled, indexing='ij')
        self.points = jnp.stack([x_upsampled_grid.ravel(), y_upsampled_grid.ravel()], axis=1)
        platform_size = int(self.inner_platform_size_in_meters * self.one_meter_length)
        self.x1 = self.hfield_half_length - (platform_size // 2)
        self.y1 = self.hfield_half_length - (platform_size // 2)
        self.x2 = self.hfield_half_length + (platform_size // 2)
        self.y2 = self.hfield_half_length + (platform_size // 2)

        root_free_joint_xml_name = env.root_free_joint_xml_name
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(root_free_joint_xml_name, env._model))

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> RoughTerrainState:
        """
        Initialize the state of the rough terrain.

        Args:
            env (Any): The environment instance.
            key (Any): JAX random key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            RoughTerrainState: The initialized terrain state.
        """
        assert_backend_is_supported(backend)
        return RoughTerrainState(backend.zeros((self.hfield_length, self.hfield_length)))

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """
        Modify the simulation specification to include the rough terrain.

        Args:
            spec (MjSpec): The simulation specification.

        Returns:
            MjSpec: The modified simulation specification.
        """
        # Create heightfield with dynamic resolution
        # MuJoCo requires file OR data, so we generate a temp PNG at runtime
        import tempfile
        from PIL import Image

        # Create a flat gray image (mid-gray = 0.5 normalized height)
        img_data = np.full((self.hfield_length, self.hfield_length), 128, dtype=np.uint8)
        img = Image.fromarray(img_data, mode='L')

        # Save to temp file
        self._temp_hfield_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(self._temp_hfield_file.name)

        spec.add_hfield(name='rough_terrain', size=self.hfield_size, file=self._temp_hfield_file.name)
        for i, field in enumerate(spec.hfields):
            if field.name == 'rough_terrain':
                self.hfield_id = i
                break

        for g in spec.geoms:
            if g.name == 'floor':
                spec.delete(g)
                break

        wb = spec.worldbody
        # Explicitly set contact parameters to match the original floor:
        # contype=1, conaffinity=1, condim=3 (friction left at default for domain randomization)
        wb.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname='rough_terrain', group=2,
                    pos=(0, 0, self.geom_z), material="MatPlane", rgba=(0.8, 0.9, 0.8, 1),
                    contype=1, conaffinity=1, condim=3)

        return spec

    def reset(self, env: Any,
              model: Union[MjModel, Model], data: Union[MjData, Data], carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the rough terrain and update its state.

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
        terrain_state = carry.terrain_state

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            height_field_raw = self.isaac_hf_to_mujoco_hf(self._jnp_random_uniform_terrain(_k), backend)
            carry = carry.replace(key=key)
        else:
            height_field_raw = self.isaac_hf_to_mujoco_hf(self._np_random_uniform_terrain(), backend)

        terrain_state = terrain_state.replace(height_field_raw=height_field_raw)
        carry = carry.replace(terrain_state=terrain_state)

        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the rough terrain and simulation state.

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
        terrain_state = carry.terrain_state
        model = self._set_attribute_in_model(model, "hfield_data", terrain_state.height_field_raw, backend)
        data = self._reset_on_edge(data, backend)
        return model, data, carry

    def get_height_matrix(self, matrix_config: Dict[str, Any],
                          env: Any,
                          model: Union[MjModel, Model],
                          data: Union[MjData, Data],
                          carry: Any,
                          backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        """
        Get the height matrix of the terrain.

        Args:
            matrix_config (Dict[str, Any]): The matrix configuration.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[np.ndarray, jnp.ndarray]: The height matrix of the terrain.

        Raises:
            NotImplementedError: Use sample_heights_at_points instead.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError("Use sample_heights_at_points instead")

    def sample_heights_at_points(self,
                                  x: Union[np.ndarray, jnp.ndarray],
                                  y: Union[np.ndarray, jnp.ndarray],
                                  model: Union[MjModel, Model],
                                  carry: Any,
                                  backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        """
        Sample terrain heights at given world (x, y) coordinates.

        Uses bilinear interpolation for smooth height values between grid points.

        Args:
            x: Array of x coordinates in world frame.
            y: Array of y coordinates in world frame.
            model: The simulation model.
            carry: Carry instance with terrain_state.
            backend: Backend module (numpy or jax.numpy).

        Returns:
            Array of terrain heights at the sample points in world frame.
        """
        assert_backend_is_supported(backend)

        terrain_state = carry.terrain_state
        height_field_raw = terrain_state.height_field_raw

        # Height field is stored FLATTENED - reshape to 2D
        # Note: terrain generation uses indexing='ij', so array is [x_idx, y_idx]
        height_field = height_field_raw.reshape(self.hfield_length, self.hfield_length)

        # World coords to hfield indices
        # Terrain spans [-hfield_half_length_in_meters, +hfield_half_length_in_meters]
        x_norm = (x + self.hfield_half_length_in_meters) / (2 * self.hfield_half_length_in_meters)
        y_norm = (y + self.hfield_half_length_in_meters) / (2 * self.hfield_half_length_in_meters)

        # Convert to indices and clamp to valid range
        x_idx = backend.clip(x_norm * (self.hfield_length - 1), 0, self.hfield_length - 1)
        y_idx = backend.clip(y_norm * (self.hfield_length - 1), 0, self.hfield_length - 1)

        # Bilinear interpolation
        x_floor = backend.floor(x_idx).astype(backend.int32)
        y_floor = backend.floor(y_idx).astype(backend.int32)
        x_ceil = backend.minimum(x_floor + 1, self.hfield_length - 1)
        y_ceil = backend.minimum(y_floor + 1, self.hfield_length - 1)

        x_frac = x_idx - x_floor
        y_frac = y_idx - y_floor

        # Sample four corners for bilinear interpolation
        # Using [x, y] indexing to match terrain generation (indexing='ij')
        h00 = height_field[x_floor, y_floor]
        h01 = height_field[x_floor, y_ceil]
        h10 = height_field[x_ceil, y_floor]
        h11 = height_field[x_ceil, y_ceil]

        # Bilinear interpolation
        h0 = h00 * (1 - y_frac) + h01 * y_frac
        h1 = h10 * (1 - y_frac) + h11 * y_frac
        heights_normalized = h0 * (1 - x_frac) + h1 * x_frac

        # Convert normalized values to actual world heights
        # MuJoCo heightfield surface formula: z_surface = geom_pos.z + hfield_data * elevation_z
        # See: https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-hfield
        heights = self.geom_z + heights_normalized * self.elevation_z

        return heights

    def isaac_hf_to_mujoco_hf(self,
                              isaac_hf: Union[np.ndarray, jnp.ndarray],
                              backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        """
        Convert Isaac height field data to MuJoCo-compatible height field data.

        Args:
            isaac_hf (Union[np.ndarray, jnp.ndarray]): The Isaac height field data.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[np.ndarray, jnp.ndarray]: The converted height field data.
        """
        assert_backend_is_supported(backend)

        hf = isaac_hf + backend.abs(backend.min(isaac_hf))
        hf /= self.mujoco_height_scaling
        return hf.reshape(-1)

    def _np_random_uniform_terrain(self) -> np.ndarray:
        """
        Generate random uniform terrain using NumPy.

        Returns:
            np.ndarray: The generated height field.
        """
        add_height_field_downsampled = (
            np.random.choice(self.heights_range, size=(int(self.hfield_length * self.random_downsampled_scale),
                                                       int(self.hfield_length * self.random_downsampled_scale))))
        interpolator = np_scipy.interpolate.RegularGridInterpolator((self.x, self.y),
                                                                    add_height_field_downsampled, method='linear')
        add_height_field = interpolator(self.points).reshape((self.hfield_length, self.hfield_length))
        add_height_field[self.x1:self.x2, self.y1:self.y2] = 0
        return add_height_field

    def _jnp_random_uniform_terrain(self, key: Any) -> jnp.ndarray:
        """
        Generate random uniform terrain using JAX.

        Args:
            key (Any): JAX random key.

        Returns:
            jnp.ndarray: The generated height field.
        """
        add_height_field_downsampled = (
            jax.random.choice(key, self.heights_range, shape=(int(self.hfield_length * self.random_downsampled_scale),
                                                              int(self.hfield_length * self.random_downsampled_scale))))
        interpolator = jnp_scipy.interpolate.RegularGridInterpolator((self.x, self.y),
                                                                     add_height_field_downsampled, method='linear')
        add_height_field = interpolator(self.points).reshape((self.hfield_length, self.hfield_length))
        add_height_field = add_height_field.at[self.x1:self.x2, self.y1:self.y2].set(0)
        return add_height_field

    def _reset_on_edge(self, data: Union[MjData, Data],
                       backend: ModuleType) -> Union[MjData, Data]:
        """
        Reset the robot position if it is on the edge of the terrain.

        Args:
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjData, Data]: The updated simulation data.
        """
        assert_backend_is_supported(backend)

        min_edge = self.hfield_half_length_in_meters - 0.5
        max_edge = self.hfield_half_length_in_meters
        com_pos = data.qpos[self._free_jnt_qpos_id][:2]
        reached_edge = backend.array(((min_edge < backend.abs(com_pos[0])) & (backend.abs(com_pos[0]) < max_edge)) | (
                    (min_edge < backend.abs(com_pos[1])) & (backend.abs(com_pos[1]) < max_edge)))
        free_jnt_xy = self._free_jnt_qpos_id[:2]
        if backend == jnp:
            init_data = data.replace(qpos=data.qpos.at[free_jnt_xy].set(0.0))
            data = jax.lax.cond(reached_edge, lambda _: init_data, lambda _: data, None)
        else:
            if reached_edge:
                data.qpos[free_jnt_xy] = 0.0

        return data
