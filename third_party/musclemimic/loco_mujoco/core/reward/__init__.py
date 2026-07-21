from .base import Reward
from .default import NoReward, TargetVelocityGoalReward, TargetXVelocityReward, LocomotionReward
from .utils import *

# register all rewards
NoReward.register()
TargetVelocityGoalReward.register()
TargetXVelocityReward.register()
LocomotionReward.register()
