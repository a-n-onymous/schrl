from typing import Callable

import numpy as np
from gym.spaces import Box

from schrl.envs import Point


class OptionPointEnv(Point):
    """
    Option environment matching with
    https://github.com/RodrigoToroIcarte/reward_machines
    and
    https://github.com/keyshor/dirl
    """

    def __init__(self, config=None):
        super(OptionPointEnv, self).__init__(config)
        self.reward_fn = None
        self.pos_mask = np.ones(14, dtype=bool)
        self.pos_mask[[3, 4]] = False

        self.observation_space = Box(-np.inf, np.inf, (12,), dtype=np.float32)

    def _ctrl_reward(self) -> float:
        assert self.reward_fn is not None, "call set_reward_fn first"
        return self.reward_fn(self.get_state())

    def set_reward_fn(self, reward_fn: Callable[[np.ndarray], float]):
        """
        Set reward function for options
        """
        self.reward_fn = reward_fn

    def get_obs(self) -> np.ndarray:
        obs = super(OptionPointEnv, self).get_obs()
        # mask out position as general practice in mujoco locomotion robots (only keep qpos and qvel)
        return obs[self.pos_mask]

    def reset(self, state: np.ndarray = None) -> np.ndarray:
        self.reward_fn.reset()
        return super(OptionPointEnv, self).reset(state)
