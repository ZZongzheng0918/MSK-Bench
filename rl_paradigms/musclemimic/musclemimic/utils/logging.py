import logging
import sys

class IdentifierAdapter(logging.LoggerAdapter):
    """Attach a stable identifier prefix to every emitted message."""

    def process(self, msg, kwargs):
        identifier = self.extra.get("identifier", "")
        if identifier and not str(msg).startswith(identifier):
            msg = f"{identifier} {msg}"
        return msg, kwargs


def setup_logger(name, level=logging.INFO, identifier="[MuscleMimic]"):
    """
    Create and return a configured logger adapter.

    When running under Hydra, messages propagate to Hydra's root logger
    (which handles file + console output). When running standalone (e.g.,
    retargeting scripts), a console handler is added.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        identifier (str): Identifier to prepend to all log messages.

    Returns:
        logging.LoggerAdapter: Configured logger adapter.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, stream=sys.stdout, format="%(levelname)s: %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return IdentifierAdapter(logger, {"identifier": identifier})


class TimestepTracker:
    """Tracks true timesteps across int32 boundaries in JAX."""
    
    def __init__(self):
        self.true_timestep = 0
        self.last_raw_step = 0
        self.wrap_count = 0
    
    def update(self, jax_timestep):
        """Update tracker with new JAX timestep and return true timestep value."""
        python_timestep = int(jax_timestep)
        
        # Convert negative to positive in uint32 space for comparison
        uint32_step = python_timestep + 2**32 if python_timestep < 0 else python_timestep
        
        # Handle wrap-around and reset cases
        if python_timestep == 0:
            if self.last_raw_step >= 2**31 or self.last_raw_step <= -2**31:
                # If last step was near the boundary, this is a wrap-around
                self.wrap_count += 1
                self.true_timestep = self.wrap_count * 2**32
            else:
                # Otherwise it's a true reset
                self.wrap_count = 0
                self.true_timestep = 0
        else:
            # Handle normal progression
            if uint32_step < self.last_raw_step:
                # We've wrapped around
                self.wrap_count += 1
            
            # Calculate true timestep
            if python_timestep < 0:
                # We're in overflow territory
                self.true_timestep = (self.wrap_count * 2**32) + uint32_step
            else:
                # We're in normal territory
                self.true_timestep = (self.wrap_count * 2**32) + python_timestep
        
        # Store current raw step for next comparison
        self.last_raw_step = uint32_step
        
        return self.true_timestep
