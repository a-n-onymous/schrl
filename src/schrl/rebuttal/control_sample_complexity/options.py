import typing

import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from schrl.config import DATA_ROOT
from schrl.rebuttal.control_sample_complexity.env import OptionPointEnv


class NSEWOptionRewardFn:
    def __init__(self, option: str):
        self.option = option

        self._proj_vec = self._get_proj_vec()
        self._prev_state = None
        self._scale = 10

    def _get_proj_vec(self) -> np.ndarray:
        if self.option == "north":
            proj_vec = np.array([0.0, 1.0])
        elif self.option == "south":
            proj_vec = np.array([0.0, -1.0])
        elif self.option == "east":
            proj_vec = np.array([1.0, 0.0])
        elif self.option == "west":
            proj_vec = np.array([-1.0, 0.0])
        else:
            raise NotImplementedError(f"unsupported option {self.option}")

        return proj_vec

    def reset(self):
        self._prev_state = None

    def __call__(self, state: np.ndarray[typing.Any, np.dtype[np.float64]]) -> float:
        if self._prev_state is not None:
            vel = state - self._prev_state
            self._prev_state = state
            return vel.dot(self._proj_vec) * self._scale
        else:
            self._prev_state = state
            return 0.0


def train_point_option_policy(option: str,
                              n_envs: int = 4,
                              total_timesteps: int = 500_000,
                              dummy_vec_env=False):
    res_path = f"{DATA_ROOT}/rebuttal/ctrl_sample_complexity"

    def env_fn():
        _env = OptionPointEnv(config={"continue_goal": True})
        _env.set_reward_fn(NSEWOptionRewardFn(option))
        init_low = np.array([-2, -2], dtype=np.float32)
        init_high = np.array([2, 2], dtype=np.float32)
        _env.update_initial_space(init_low, init_high)
        return Monitor(TimeLimit(_env, 1000))

    vec_env_cls = DummyVecEnv if dummy_vec_env else SubprocVecEnv
    vec_env = vec_env_cls(env_fns=[env_fn] * n_envs)

    algo = PPO("MlpPolicy", vec_env,
               tensorboard_log=f"{res_path}/logs/", verbose=1)
    algo.learn(total_timesteps, tb_log_name=f"{option}_pi")

    algo.save(f"{res_path}/models/point_{option}_policy.zip")
