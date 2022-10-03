from .wrapper import DronePIDEnv, Car, Doggo, Point, GoalEnv


def make_env(env_name: str, enable_gui: bool = False) -> GoalEnv:
    if env_name == "drone":
        return DronePIDEnv(gui=enable_gui)
    elif env_name == "point":
        return Point()
    elif env_name == "car":
        return Car()
    elif env_name == "doggo":
        return Doggo()
    else:
        raise NotImplementedError(f"{env_name} is not supported yet")
