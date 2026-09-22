"""Minimal Viser-based viewer for MuscleMimic MuJoCo environments.

Scope (MVP):
- MuJoCo CPU environments only (no MJX batching yet)
- Visual geoms per body merged into a single mesh and streamed via Viser
- Basic controls: play/pause, reset, speed +/-

Deliberately out-of-scope for MVP:
- Ghost robot visualization (goal visuals)
- Contact point/force visualization
- Recording/export via Viser
"""

from __future__ import annotations

import time
from typing import Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np


class ViserViewer:
    def __init__(
        self,
        env,
        agent_conf,
        agent_state,
        deterministic: bool = True,
        frame_rate: float = 60.0,
        include_collision: bool = False,
    ) -> None:
        """Construct a viewer.

        Args:
            env: MuJoCo environment instance (CPU, not MJX)
            agent_conf: PPOJax agent configuration (network + config)
            agent_state: PPOJax agent state with trained params
            deterministic: If True, set log_std to -inf for deterministic actions
            frame_rate: Target FPS for visualization
            include_collision: If True, display collision geoms instead of visual geoms
        """
        self.env = env
        self.agent_conf = agent_conf
        self.agent_state = agent_state
        self.deterministic = deterministic
        self.frame_rate = frame_rate
        self.include_collision = include_collision

        # Initialized during setup
        self._server = None
        self._handles = None
        self._tendon_handle = None
        self._tendon_scene = None
        self._tendon_option = None
        self._tendon_camera = None
        self._paused = False
        self._time_multiplier = 1.0

    def _prepare_policy(self):
        """Create a jitted policy call matching MuscleMimic eval semantics."""

        def sample_actions(ts, obs, _rng):
            if hasattr(obs, "ndim") and obs.ndim == 1:
                obs_b = jnp.atleast_2d(obs)
            else:
                obs_b = obs
            vars_in = {"params": ts.params, "run_stats": ts.run_stats}
            y, updates = self.agent_conf.network.apply(vars_in, obs_b, mutable=["run_stats"])
            pi, _ = y
            ts_out = ts.replace(run_stats=updates["run_stats"])  # update stats
            a = pi.sample(seed=_rng)
            if hasattr(a, "ndim") and a.ndim > 1 and a.shape[0] == 1:
                a = a[0]
            return a, ts_out

        train_state = self.agent_state.train_state
        if self.deterministic:
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        # Handle multi-seed agent state if present
        config = self.agent_conf.config.experiment
        if getattr(config, "n_seeds", 1) > 1:
            # Default to seed 0 for viewer
            train_state = jax.tree.map(lambda x: x[0], train_state)

        rng = jax.random.key(0)
        plcy_call = jax.jit(sample_actions)
        return plcy_call, train_state, rng

    def _setup_viser(self, model: mujoco.MjModel):
        try:
            import viser  # type: ignore
            import viser.transforms as vtf  # used for type import completeness
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Viser is not installed. Install optional extras with: pip install '.[viser]'") from e

        # Lazy import trimesh helpers
        from .viser_utils import build_body_meshes

        self._server = viser.ViserServer(label="musclemimic")
        self._server.scene.configure_environment_map(environment_intensity=0.8)

        # GUI controls
        tabs = self._server.gui.add_tab_group()
        with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
            self._status_html = self._server.gui.add_html("")
            # Simulation
            with self._server.gui.add_folder("Simulation"):
                self._pause_button = self._server.gui.add_button(
                    "Play" if self._paused else "Pause",
                    icon=viser.Icon.PLAYER_PLAY if self._paused else viser.Icon.PLAYER_PAUSE,
                )

                @self._pause_button.on_click
                def _(_ev) -> None:
                    self._paused = not self._paused
                    self._pause_button.label = "Play" if self._paused else "Pause"
                    self._pause_button.icon = viser.Icon.PLAYER_PLAY if self._paused else viser.Icon.PLAYER_PAUSE
                    self._update_status()

                reset_button = self._server.gui.add_button("Reset Environment")

                @reset_button.on_click
                def _(_ev) -> None:
                    self.env.reset()
                    mujoco.mj_forward(model, self._get_data())
                    self._sync_meshes(model, self._get_data())
                    self._update_status()

                speed_buttons = self._server.gui.add_button_group("Speed", options=["Slower", "Faster"])

                @speed_buttons.on_click
                def _(event) -> None:
                    if event.target.value == "Slower":
                        self._time_multiplier = max(0.1, self._time_multiplier / 2.0)
                    else:
                        self._time_multiplier = min(4.0, self._time_multiplier * 2.0)
                    self._update_status()

        # Add ground plane grid
        self._server.scene.add_grid(
            "/ground",
            width=10.0,
            height=10.0,
            width_segments=20,
            height_segments=20,
            plane="xy",
            cell_color=(180, 180, 180),
            section_color=(120, 120, 120),
            cell_thickness=1,
            section_thickness=2,
        )

        # Build body meshes
        body_meshes = build_body_meshes(model, include_collision=self.include_collision)
        handles = {}
        with self._server.atomic():
            for body_id, mesh in body_meshes.items():
                # Use batched handle API even for batch=1 for future extension
                handle = self._server.scene.add_batched_meshes_trimesh(
                    f"/bodies/{body_id}",
                    mesh,
                    batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]]),
                    batched_positions=np.array([[0.0, 0.0, 0.0]]),
                    lod="off",
                    visible=True,
                )
                handles[body_id] = handle
        self._handles = handles
        self._setup_tendon_visuals(model)
        self._update_status()

    def _setup_tendon_visuals(self, model: mujoco.MjModel) -> None:
        if model.ntendon == 0:
            return
        maxgeom = max(10000, model.ngeom + model.ntendon * 20)
        self._tendon_scene = mujoco.MjvScene(model, maxgeom=maxgeom)
        self._tendon_option = mujoco.MjvOption()
        self._tendon_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1
        self._tendon_camera = mujoco.MjvCamera()
        points, colors = self._build_tendon_segments(model, self._get_data())
        if points.size == 0:
            return
        self._tendon_handle = self._server.scene.add_line_segments(
            "/tendons",
            points=points,
            colors=colors,
            line_width=3,
            visible=True,
        )

    def _build_tendon_segments(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._tendon_scene is None or self._tendon_option is None or self._tendon_camera is None:
            return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.uint8)
        mujoco.mjv_updateScene(
            model,
            data,
            self._tendon_option,
            None,
            self._tendon_camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._tendon_scene,
        )

        # Vectorized extraction of tendon geoms
        ngeom = self._tendon_scene.ngeom
        if ngeom == 0:
            return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.uint8)

        geoms = self._tendon_scene.geoms[:ngeom]

        # Filter for tendon geoms only (vectorized)
        objtype_arr = np.array([g.objtype for g in geoms], dtype=np.int32)
        tendon_mask = objtype_arr == mujoco.mjtObj.mjOBJ_TENDON
        tendon_indices = np.where(tendon_mask)[0]

        if len(tendon_indices) == 0:
            return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.uint8)

        # Extract data for tendon geoms (batch extraction)
        n_tendons = len(tendon_indices)
        positions = np.empty((n_tendons, 3), dtype=np.float32)
        axes = np.empty((n_tendons, 3), dtype=np.float32)
        halves = np.empty(n_tendons, dtype=np.float32)
        rgbas = np.empty((n_tendons, 4), dtype=np.float32)

        for i, idx in enumerate(tendon_indices):
            g = geoms[idx]
            positions[i] = g.pos
            axes[i] = g.mat[:, 2]
            halves[i] = g.size[2]
            rgbas[i] = g.rgba

        # Vectorized point calculation
        offsets = axes * halves[:, np.newaxis]
        p0 = positions - offsets
        p1 = positions + offsets
        points = np.stack([p0, p1], axis=1).astype(np.float32)

        # Vectorized color calculation
        rgbas = np.clip(rgbas, 0.0, 1.0)
        rgb = rgbas[:, :3] * 0.7 * rgbas[:, 3:4]  # darker multiplier with alpha
        colors_uint8 = (rgb * 255.0).astype(np.uint8)
        # Each segment needs color for both endpoints
        colors = np.stack([colors_uint8, colors_uint8], axis=1)

        return points, colors

    def _get_data(self) -> mujoco.MjData:
        # Navigate wrappers if any to find data
        if hasattr(self.env, "data"):
            return self.env.data
        if hasattr(self.env, "env"):
            cur = self.env
            while hasattr(cur, "env"):
                cur = cur.env
                if hasattr(cur, "data"):
                    return cur.data
        raise RuntimeError("Could not access MuJoCo data from environment")

    def _update_status(self):
        if self._server is None:
            return
        status = f"<b>Paused:</b> {self._paused} &nbsp; <b>Speed:</b> {self._time_multiplier:.2f}x"
        self._status_html.content = status

    def _sync_meshes(self, model: mujoco.MjModel, data: mujoco.MjData, update_tendons: bool = True) -> None:
        import numpy as _np

        # body_xpos/xquat are sized to nbody
        body_xpos = _np.array(data.xpos)
        body_xquat = _np.array(data.xquat)

        # Batch all updates in a single atomic context for better performance
        with self._server.atomic():
            for body_id, handle in self._handles.items():
                if body_id >= len(body_xpos):
                    continue
                pos = body_xpos[body_id]
                quat = body_xquat[body_id]  # wxyz
                handle.batched_positions = _np.array([pos], dtype=float)
                handle.batched_wxyzs = _np.array([quat], dtype=float)
            # Update tendons within same atomic context
            if update_tendons and self._tendon_handle is not None:
                points, colors = self._build_tendon_segments(model, data)
                self._tendon_handle.points = points
                self._tendon_handle.colors = colors
        # push changes to clients
        self._server.flush()

    def _sync_tendons(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._tendon_handle is None:
            return
        points, colors = self._build_tendon_segments(model, data)
        self._tendon_handle.points = points
        self._tendon_handle.colors = colors

    def run(self, n_steps: int | None = None) -> None:
        """Main loop: sample actions, step env, and stream transforms to Viser."""
        # Resolve model/data from environment
        model = self.env.model if hasattr(self.env, "model") else self.env.env.model
        data = self._get_data()

        # Prepare policy
        plcy_call, train_state, rng = self._prepare_policy()

        # Reset env and forward kinematics
        obs = self.env.reset()
        mujoco.mj_forward(model, data)

        # Set up Viser (now data has valid state for tendon visualization)
        self._setup_viser(model)
        self._sync_meshes(model, data)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        target_dt = 1.0 / float(self.frame_rate)

        step_count = 0
        try:
            while step_count < n_steps:
                t0 = time.time()

                if not self._paused:
                    # Sample and step
                    rng, _rng = jax.random.split(rng)
                    action, train_state = plcy_call(train_state, obs, _rng)
                    action = jnp.atleast_2d(action)
                    obs, _r, _abs, done, _info = self.env.step(action)

                    # Sync transforms
                    mujoco.mj_forward(model, data)
                    self._sync_meshes(model, data)

                    if done:
                        obs = self.env.reset()
                        mujoco.mj_forward(model, data)
                        self._sync_meshes(model, data)

                    step_count += 1

                # Frame pacing
                elapsed = time.time() - t0
                to_sleep = max(0.0, (target_dt / self._time_multiplier) - elapsed)
                if to_sleep > 0:
                    time.sleep(to_sleep)

        except KeyboardInterrupt:
            pass
        finally:
            # Attempt to close server cleanly
            try:
                if self._server is not None:
                    self._server.stop()
            except Exception:
                pass
