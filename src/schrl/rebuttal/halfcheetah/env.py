from typing import Dict

import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.spaces import Box

from schrl.envs.wrapper import GoalEnv


class GoalConditionedHalfCheetah(GoalEnv):
    def __init__(self, continuous_goals: bool = True):
        self.mj_env = HalfCheetahEnv()
        obs_space = Box(np.concatenate([[-20.0], self.mj_env.observation_space.low], dtype=np.float32),
                        np.concatenate([[20.0], self.mj_env.observation_space.high], dtype=np.float32))
        action_space = self.mj_env.action_space
        init_space = Box(np.array(-1.0, dtype=np.float32),
                         np.array(1.0, dtype=np.float32))
        goal_space = Box(np.array(-10.0, dtype=np.float32),
                         np.array(10.0, dtype=np.float32))
        goal = np.array(0.0)
        reach_reward = 20
        super().__init__(obs_space, action_space, init_space, goal_space, goal, reach_reward)
        self.continuous_goals = continuous_goals

    def set_state(self, state: np.ndarray):
        qpos = self.mj_env.sim.data.qpos.flat.copy()
        qvel = self.mj_env.sim.data.qvel.flat.copy()
        qpos[0] = state
        self.mj_env.set_state(qpos, qvel)

    def get_obs(self) -> np.ndarray:
        qpos = self.mj_env.sim.data.qpos.flat.copy()
        qvel = self.mj_env.sim.data.qvel.flat.copy()
        qpos[0] -= self.get_goal()
        observation = np.concatenate((qpos, qvel)).ravel()

        return observation

    def reach(self) -> bool:
        return np.abs(self.get_state() - self.get_goal()) < 0.3

    def update_state(self, action: np.ndarray) -> Dict:
        _, _, _, info = self.mj_env.step(action)

        return info

    def step(self, action):
        obs, r, done, info = super().step(action * 5)
        r -= 0.1 * self.mj_env.control_cost(action)

        if self.continuous_goals and info["reach"]:
            self.set_goal(self.goal_space.sample())

        return obs, r, done, info

    def get_state(self) -> np.ndarray:
        qpos = self.mj_env.sim.data.qpos.flat.copy()
        return np.array([qpos[0]])

    def collision(self) -> bool:
        return False

    def render(self, mode="human"):
        self.mj_env.render()
