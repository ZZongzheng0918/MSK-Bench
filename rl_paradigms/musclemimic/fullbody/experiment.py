import sys
import traceback

import hydra
import jax
from omegaconf import DictConfig

from musclemimic.runner.engine import run_experiment
from musclemimic.runner.logging import UnifiedHooks

jax.config.update("jax_default_matmul_precision", "high")


@hydra.main(version_base=None, config_path="./", config_name="conf_fullbody")
def experiment(config: DictConfig):
    try:
        run_experiment(config, hooks=UnifiedHooks())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
