from stable_baselines3.ppo import PPO

from schrl.config import DATA_ROOT
from schrl.hrl.hrl import HRL
from schrl.tasks import get_task


def load_general_ctrl_pi(robot_name: str, algo_name: str = "ppo"):
    if algo_name not in ("ppo",):
        raise NotImplementedError(f"Unsupported policy {algo_name}")
    path = f"{DATA_ROOT}/general_ctrl_pi/{robot_name}_{algo_name}.zip"
    return PPO.load(path)


def load_schrl_pi(env_name: str, task_name: str, enable_gui: bool = False):
    if env_name not in ("drone", "point", "car", "doggo"):
        raise NotImplementedError(f"Unsupported environment {env_name}")

    if task_name not in ("seq", "cover", "branch", "loop", "signal"):
        raise NotImplementedError(f"Unsupported task {task_name}")

    task = get_task(env_name, task_name, gui=enable_gui)
    model_path = f"{DATA_ROOT}/schrl_pi/pretrained/{env_name}/{task_name}/"

    return HRL.load(model_path, task)
