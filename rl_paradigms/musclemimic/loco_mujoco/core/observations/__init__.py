from .base import Observation, ObservationIndexContainer, ObservationContainer, ObservationType, StatefulObservation
from .goals import Goal, NoGoal, GoalRandomRootVelocity, GoalTrajRootVelocity

# register base goals
NoGoal.register()
GoalRandomRootVelocity.register()
GoalTrajRootVelocity.register()
