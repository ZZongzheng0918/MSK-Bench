from .base import TaskFactory
from .rl_factory import RLFactory
from .imitation_factory import ImitationFactory
from .dataset_confs import AMASSDatasetConf, LAFAN1DatasetConf, CustomDatasetConf


# register factories
RLFactory.register()
ImitationFactory.register()


