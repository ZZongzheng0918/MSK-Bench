from .base import TerminalStateHandler
from .no_terminal import NoTerminalStateHandler
from .height import HeightBasedTerminalStateHandler
from .traj import RootPoseTrajTerminalStateHandler
from .bimanual import BimanualTerminalStateHandler

# register all terminal state handlers
NoTerminalStateHandler.register()
HeightBasedTerminalStateHandler.register()
RootPoseTrajTerminalStateHandler.register()
BimanualTerminalStateHandler.register()
