import os
from typing import Union

import cloudpickle as pickle
import numpy as np
import torch as th
from gym.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import Tensor

from schrl.config import DEFAULT_DEVICE, DEFAULT_TENSOR_TYPE


def default_tensor(inpt) -> th.Tensor:
    return th.tensor(inpt,
                     dtype=DEFAULT_TENSOR_TYPE,
                     device=DEFAULT_DEVICE)


def as_numpy(inpt: Union[th.Tensor, int, float]) -> np.ndarray:
    if isinstance(inpt, Tensor):
        return inpt.detach().cpu().numpy()
    else:
        return np.array(inpt)


def pickle_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def choose_device(device: str):
    assert device in ("auto", "cpu", "cuda"), f"Unsupported device {device}"

    if device == "auto":
        if th.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device


def wrap_env(env,
             time_limit: int = 8000,
             sb3_reward_monitor: bool = True,
             vec_env: bool = True):
    if time_limit > 0:
        env = TimeLimit(env, time_limit)

    if sb3_reward_monitor:
        env = Monitor(env)

    if vec_env:
        env = DummyVecEnv([lambda: env])

    return env
