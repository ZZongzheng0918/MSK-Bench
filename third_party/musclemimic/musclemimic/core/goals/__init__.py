"""Goals module for musclemimic."""

from musclemimic.core.goals.bimanual import GoalBimanualTrajMimic, GoalBimanualTrajMimicv2
from musclemimic.core.goals.trajectory import GoalTrajMimic, GoalTrajMimicv2

# Register all trajectory mimic goals
GoalTrajMimic.register()
GoalTrajMimicv2.register()
GoalBimanualTrajMimic.register()
GoalBimanualTrajMimicv2.register()

__all__ = [
    "GoalBimanualTrajMimic",
    "GoalBimanualTrajMimicv2",
    "GoalTrajMimic",
    "GoalTrajMimicv2",
]
