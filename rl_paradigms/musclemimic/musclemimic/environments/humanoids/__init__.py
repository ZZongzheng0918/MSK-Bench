from .myofullbody import MyoFullBody, MjxMyoFullBody
from .bimanual import MyoBimanualArm, MjxMyoBimanualArm


# register muscle environments
MyoBimanualArm.register()
MjxMyoBimanualArm.register()
MyoFullBody.register()
MjxMyoFullBody.register()
