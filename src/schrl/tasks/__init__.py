from .base import TaskBase
from .car import CarSequence, CarCover, CarBranch, CarLoop, CarSignal
from .doggo import DoggoSequence, DoggoCover, DoggoBranch, DoggoLoop, DoggoSignal
from .drone import DroneSequence, DroneCover, DroneBranch, DroneLoop, DroneSignal
from .point import PointSequence, PointCover, PointBranch, PointLoop, PointSignal


def get_task(robot_name: str, task_name: str, gui: bool) -> TaskBase:
    _table = {
        "drone": {
            "seq": DroneSequence,
            "cover": DroneCover,
            "branch": DroneBranch,
            "loop": DroneLoop,
            "signal": DroneSignal
        },
        "point": {
            "seq": PointSequence,
            "cover": PointCover,
            "branch": PointBranch,
            "loop": PointLoop,
            "signal": PointSignal
        },
        "car": {
            "seq": CarSequence,
            "cover": CarCover,
            "branch": CarBranch,
            "loop": CarLoop,
            "signal": CarSignal
        },
        "doggo": {
            "seq": DoggoSequence,
            "cover": DoggoCover,
            "branch": DoggoBranch,
            "loop": DoggoLoop,
            "signal": DoggoSignal
        },

    }

    return _table[robot_name][task_name](enable_gui=gui)
