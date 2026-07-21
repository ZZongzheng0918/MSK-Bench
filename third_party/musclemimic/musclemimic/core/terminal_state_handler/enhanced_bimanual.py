from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import MjData
from mujoco.mjx import Data

from loco_mujoco.core.terminal_state_handler.bimanual import (
    BimanualTerminalStateHandler,
)
from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__, identifier="[BimanualTerminal]")


class EnhancedBimanualTerminalStateHandler(BimanualTerminalStateHandler):
    """
    Terminal state handler for bimanual motion imitation tasks.

    Episodes terminate when the tracked mimic sites deviate too far from the
    current reference trajectory frame.
    """

    def __init__(
        self,
        env: Any,
        max_site_deviation: float = 0.5,
        enable_reference_check: bool = True,
        site_deviation_mode: str = "mean",
        # Debug controls (use jax.debug.print inside JIT)
        debug_print: bool = False,
        debug_every: int = 5000,
        **handler_config: dict[str, Any],
    ):
        """
        Initialize enhanced bimanual terminal state handler.

        Args:
            env: The environment instance
            max_site_deviation: Maximum site position deviation (meters)
            enable_reference_check: Whether to check reference tracking
            site_deviation_mode: "max" (any site) or "mean" (average across sites)
            debug_print: Enable periodic debug output for the JAX path
            debug_every: Debug print interval in environment steps
            **handler_config: Additional configuration passed to parent
        """
        super().__init__(env, **handler_config)

        # Debug options
        self._debug_print = bool(debug_print)
        self._debug_every = int(debug_every)

        # Site-based termination parameters
        self.max_site_deviation = max_site_deviation
        self.enable_reference_check = enable_reference_check

        # Validate and store deviation mode
        valid_modes = ["max", "mean"]
        if site_deviation_mode not in valid_modes:
            raise ValueError(f"site_deviation_mode must be one of {valid_modes}")
        self.site_deviation_mode = site_deviation_mode

        logger.info("EnhancedBimanualTerminalStateHandler initialized")
        logger.info("  Max site deviation: %sm", self.max_site_deviation)
        logger.info("  Site deviation mode: %s", self.site_deviation_mode)
        logger.info("  Reference check enabled: %s", self.enable_reference_check)
        if self._debug_print:
            logger.info("  Debug prints enabled every %s steps (jax.debug.print)", self._debug_every)

    def _check_site_deviations(self, env: Any, state: Any, carry: Any, backend: ModuleType) -> bool | jnp.ndarray:
        """
        Compare tracked mimic sites against the current reference trajectory frame.
        """
        # Exit early when trajectory or site data is unavailable.
        if not self.enable_reference_check:
            return backend.array(False)

        if not hasattr(env, "th") or env.th is None:
            return backend.array(False)

        if not (hasattr(env, "_goal") and hasattr(env._goal, "_rel_site_ids")):
            return backend.array(False)

        # Early return if no site data
        if not hasattr(state, "site_xpos"):
            return backend.array(False)

        # Reference data for the current trajectory step.
        ref_data = env.th.get_current_traj_data(carry, backend)

        # No site data means there is nothing to compare against.
        if not hasattr(ref_data, "site_xpos"):
            return backend.array(False)

        # Goal stores the model site ids used for mimic tracking.
        site_mapping = env._goal._rel_site_ids

        # Extract the tracked sites from the full simulation state.
        current_mapped_sites = state.site_xpos[site_mapping]

        # Trajectory data may store only a reduced site set, so resolve the
        # tracked model ids into trajectory indices when needed.
        if env._goal._site_mapper.requires_mapping:
            # Resolve indices explicitly so reduced trajectory caches cannot be
            # indexed with raw model site ids by mistake.
            try:
                traj_indices = env._goal._site_mapper.model_ids_to_traj_indices(site_mapping)
            except Exception as e:
                # Mapping not available or invalid: disable termination this step but only
                # emit JAX-side debug output when explicit debug printing is enabled.
                if backend == jnp and self._debug_print:
                    jax.debug.print("[EnhancedBimanualTS] Site mapping failed: {}", str(e))
                elif backend != jnp:
                    logger.warning("[EnhancedBimanualTS] Site mapping failed: %s", e)
                return backend.array(False)
            ref_mapped_sites = ref_data.site_xpos[traj_indices]
        else:
            ref_mapped_sites = ref_data.site_xpos

        # Compute per-site Euclidean errors.
        site_deviations = backend.linalg.norm(current_mapped_sites - ref_mapped_sites, axis=-1)

        # Apply either a max-site or mean-site threshold.
        violations = backend.greater(site_deviations, self.max_site_deviation)

        is_max_mode = self.site_deviation_mode == "max"

        max_violation = backend.any(violations)

        mean_deviation = backend.mean(site_deviations)
        mean_violation = backend.greater(mean_deviation, self.max_site_deviation)

        # Select the configured reduction without Python branching.
        result = backend.where(is_max_mode, max_violation, mean_violation)

        # Optional debug logging for long JAX runs.
        if self._debug_print:
            step_no = getattr(carry, "cur_step_in_episode", 0)
            if backend == jnp:
                # Gate printing using JAX predicates to avoid Python control flow on tracers.
                per = jnp.equal(jnp.remainder(step_no, jnp.asarray(self._debug_every, dtype=step_no.dtype)), 0)
                do_print = jnp.logical_or(result, jnp.logical_and(jnp.asarray(self._debug_every > 0), per))

                def _print_fn(_):
                    jax.debug.print(
                        "[EnhancedBimanualTS] step={} mean_dev={:.3f} thresh={} mode={} violation={}",
                        step_no,
                        mean_deviation,
                        self.max_site_deviation,
                        self.site_deviation_mode,
                        result,
                    )
                    return jnp.int32(0)

                _ = jax.lax.cond(do_print, _print_fn, lambda _: jnp.int32(0), operand=None)
            else:
                if (self._debug_every > 0 and (int(step_no) % self._debug_every == 0)) or bool(result):
                    logger.debug(
                        "[EnhancedBimanualTS] step=%s mean_dev=%.3f thresh=%s mode=%s violation=%s",
                        int(step_no),
                        float(mean_deviation),
                        self.max_site_deviation,
                        self.site_deviation_mode,
                        bool(result),
                    )

        return result

    def is_absorbing(self, env: Any, obs: np.ndarray, info: dict[str, Any], data: MjData, carry: Any) -> bool | Any:
        """
        Check if the current state is terminal (CPU Mujoco version).

        Args:
            env: The environment instance
            obs: Observations
            info: Info dictionary
            data: Mujoco data structure
            carry: Additional carry information

        Returns:
            Tuple of (is_absorbing, carry)
        """
        return self._is_absorbing_compat(env, obs, info, data, carry, np)

    def mjx_is_absorbing(self, env: Any, obs: jnp.ndarray, info: dict[str, Any], data: Data, carry: Any) -> bool | Any:
        """
        Check if the current state is terminal (Mjx version).

        Args:
            env: The environment instance
            obs: Observations
            info: Info dictionary
            data: Mjx data structure
            carry: Additional carry information

        Returns:
            Tuple of (is_absorbing, carry)
        """
        return self._is_absorbing_compat(env, obs, info, data, carry, jnp)

    def _is_absorbing_compat(
        self,
        env: Any,
        obs: np.ndarray | jnp.ndarray,
        info: dict[str, Any],
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> bool | Any:
        """
        Shared termination path for both numpy and JAX backends.

        Args:
            env: The environment instance
            obs: Observations
            info: Info dictionary
            data: Simulation data
            carry: Additional carry information
            backend: Backend module (numpy or jax.numpy)

        Returns:
            Tuple of (should_terminate, carry)
        """
        # Bimanual termination currently depends only on reference tracking quality.
        site_violation = self._check_site_deviations(env, data, carry, backend)

        should_terminate = site_violation

        return should_terminate, carry


# Register handler on import for availability via string
EnhancedBimanualTerminalStateHandler.register()
