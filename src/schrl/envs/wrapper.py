from abc import ABC, abstractmethod
from typing import Dict

import gym
import numpy as np
import pybullet as p
from gym.spaces import Box

from schrl.envs.base import BulletEnv
from schrl.envs.mujoco_robots.robots.engine import Engine
from schrl.envs.pybullet_robots.control import DronePID
from schrl.envs.pybullet_robots.env_editor import EnvEditor
from schrl.envs.pybullet_robots.robots.drone import Drone


def make_pybullet_env(robot_name: str, enable_gui=True) -> BulletEnv:
    if robot_name == "drone":
        robot = Drone(enable_gui=enable_gui)
        return BulletEnv(robot)

    raise NotImplementedError(f"{robot_name}")


class GoalEnv(gym.Env, ABC):
    def __init__(self,
                 observation_space: Box,
                 action_space: Box,
                 init_space: Box,
                 goal_space: Box,
                 goal: np.ndarray,
                 reach_reward: float):
        self.observation_space = observation_space
        self.action_space = action_space
        self.init_space = init_space
        self.goal_space = goal_space
        self._goal = goal
        self.reach_reward = reach_reward

        self.prev_dist_to_goal = None
        self.init_state = None
        self._goal_space_size = None
        self._stop = False

    @abstractmethod
    def set_state(self, state: np.ndarray):
        pass

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def reach(self) -> bool:
        pass

    @abstractmethod
    def update_state(self, action: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def collision(self) -> bool:
        pass

    def sample_init_state(self) -> np.ndarray:
        return self.init_space.sample()

    def reset(self, state: np.ndarray = None) -> np.ndarray:
        self.prev_dist_to_goal = None
        self._stop = False
        if state is None:
            state = self.sample_init_state()
        self.set_state(state)
        self.init_state = self.get_state()

        return self.get_obs()

    def step(self, action: np.ndarray):
        info = self.update_state(action)
        reach = self.reach()
        if reach:
            reward = self.reach_reward
        else:
            reward = self._ctrl_reward()

        info["reach"] = reach
        info["collision"] = self.collision()
        done = info["collision"] or self._stop

        return self.get_obs(), reward, done, info

    def set_goal(self, goal: np.ndarray):
        self._goal = goal
        self.prev_dist_to_goal = None

    def get_goal(self) -> np.ndarray:
        return self._goal

    def _ctrl_reward(self) -> float:
        """
        a ``stateful'' reward function
        """
        dist_to_goal = np.linalg.norm(self.get_goal() - self.get_state())

        if self.prev_dist_to_goal is None or self.reach():
            self.prev_dist_to_goal = dist_to_goal
            return 0.0

        disp = self.prev_dist_to_goal - dist_to_goal
        reward = disp  # positive if closer to goal

        self.prev_dist_to_goal = dist_to_goal
        return reward

    @property
    def goal_space_size(self) -> int:
        if self._goal_space_size is None:
            self._goal_space_size = len(self.sample_init_state())
        return self._goal_space_size

    def stop_in_next_step(self):
        self._stop = True


class DroneEnv(GoalEnv):

    def __init__(self, gui=False, reset_goal=False):
        self.drone_env = make_pybullet_env("drone", gui)
        self.gui = gui
        self.reset_goal = reset_goal

        reach_reward = 20.0
        goal = np.array([0, 0, 5])

        goal_low = np.array([-3, -3, 1], dtype=np.float32)
        goal_high = np.array([3, 3, 3], dtype=np.float32)
        goal_space = Box(low=goal_low, high=goal_high)

        init_low = np.array([-5, 4, 3], dtype=np.float32)
        init_high = np.array([-4, 5, 8], dtype=np.float32)
        init_space = Box(low=init_low, high=init_high)

        obs_low = np.array([-5, -5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                           dtype=np.float32)
        obs_high = np.array(
            [5, 5, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        observation_space = Box(low=obs_low, high=obs_high)

        action_low = np.array([-1, -1, -1, -1], dtype=np.float32)
        action_high = np.array([1, 1, 1, 1], dtype=np.float32)
        action_space = Box(low=action_low, high=action_high)

        super(DroneEnv, self).__init__(observation_space,
                                       action_space, init_space, goal_space, goal, reach_reward)

        self.robot: Drone = self.drone_env.robot  # type: ignore
        self.robot_id = self.robot.robot_id
        self.client_id = self.drone_env.client_id
        _, self.ori = p.getBasePositionAndOrientation(
            self.robot_id, self.client_id)

        self.env_editor = EnvEditor(self.robot.client_id)
        self._prev_goal_id = None
        self.set_goal(self.get_goal(), plot=False)

        self._reach = False

    def reset(self, state: np.ndarray = None) -> np.ndarray:
        if self.reset_goal:
            new_goal = self.goal_space.sample()
            self.set_goal(new_goal)
        self._reach = False
        return super(DroneEnv, self).reset(state)

    def set_goal(self, goal: np.ndarray, plot=True):
        if plot:
            if self._prev_goal_id is not None:
                self.env_editor.remove_body(self._prev_goal_id)
            self._prev_goal_id = self.env_editor.add_ball(goal)
        super(DroneEnv, self).set_goal(goal)

    def set_state(self, state: np.ndarray):
        p.resetBasePositionAndOrientation(
            self.robot_id, state, self.ori, self.client_id)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [
                            0, 0, 0], physicsClientId=self.client_id)

    def get_state(self) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(self.robot_id, self.client_id)[0])

    def _ctrl_reward(self) -> float:
        reward = float(super(DroneEnv, self)._ctrl_reward())
        return reward

    def update_state(self, action: np.ndarray):
        action = action.clip(-1, 1) ** 5
        action += self.robot.hover_rpm / self.robot.max_rpm
        action = action.clip(0, 1)
        self.drone_env.step(action * self.robot.max_rpm)

        self._reach = np.linalg.norm(self.get_goal() - self.get_state()) < 0.3
        if self.reset_goal and self.reach():
            new_goal = self.goal_space.sample()
            self.set_goal(new_goal)

        return {"collision": self.collision()}

    def collision(self) -> bool:
        return False

    def get_obs(self) -> np.ndarray:
        abs_obs = self.robot.get_obs()
        obs = abs_obs
        obs[:3] = obs[:3] - self.get_goal()

        return obs

    def reach(self) -> bool:
        return self._reach

    def render(self, mode="human"):
        assert self.gui, "must enable gui for render"

    def close(self):
        p.resetSimulation()
        p.disconnect(self.drone_env.client_id)


class DronePIDEnv(DroneEnv):
    def __init__(self, gui=False):
        super(DronePIDEnv, self).__init__(gui)
        action_space_high = np.ones(18, dtype=np.float32)
        action_space_low = -np.ones(18, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=action_space_low, high=action_space_high)

        self.coef_mean = np.array([.1, .1, .2,
                                   .0001, .0001, .0001,
                                   .1, .1, .1,
                                   .3, .3, .05,
                                   .0001, .0001, .0001,
                                   .3, .3, .5]) * 2
        self.coef_radius = np.array([.1, .1, .2,
                                     .0001, .0001, .0001,
                                     .1, .1, .1,
                                     .3, .3, .05,
                                     .0001, .0001, .0001,
                                     .3, .3, .5])

        self.pid_controller = DronePID(self.drone_env.robot)  # type: ignore

    def reset(self, state: np.ndarray = None) -> np.ndarray:
        self.pid_controller.reset()
        return super(DronePIDEnv, self).reset(state)

    def update_state(self, action: np.ndarray):
        coefs = self.coef_mean + action * self.coef_radius
        _force_p_coef = coefs[0:3]
        _force_i_coef = coefs[3:6]
        _force_d_coef = coefs[6:9]
        _torque_p_coef = coefs[9:12]
        _torque_i_coef = coefs[12:15]
        _torque_d_coef = coefs[15:18]
        self.pid_controller.set_force_pid_coef(
            _force_p_coef, _force_i_coef, _force_d_coef)
        self.pid_controller.set_torque_pid_coef(
            _torque_p_coef, _torque_i_coef, _torque_d_coef)

        self._reach = np.linalg.norm(self.get_goal() - self.get_state()) < 0.6
        if self.reset_goal and self.reach():
            new_goal = self.goal_space.sample()
            self.set_goal(new_goal)

        cmd = self.pid_controller.control(self.get_goal())
        self.drone_env.step(cmd)

        return {"collision": self.collision()}


class MjEnvBase(GoalEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}

    BASE_SENSORS = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    EXTRA_SENSORS = {
        'doggo': [
            'touch_ankle_1a',
            'touch_ankle_2a',
            'touch_ankle_3a',
            'touch_ankle_4a',
            'touch_ankle_1b',
            'touch_ankle_2b',
            'touch_ankle_3b',
            'touch_ankle_4b'
        ],
    }
    ROBOT_OVERRIDES = {
        'car': {
            'box_size': 0.125,  # Box half-radius size
            'box_keepout': 0.125,  # Box keepout radius for placement
            'box_density': 0.0005,
        },
    }

    def __init__(self,
                 robot_name: str = "point",
                 config: Dict = None,
                 observe_abs_pos: bool = False):
        base_config = {
            "robot_base": f"xmls/{robot_name}.xml",
            'sensors_obs': self.BASE_SENSORS,
            'observe_com': observe_abs_pos,
            'observe_goal_comp': not observe_abs_pos
        }
        if config is None:
            config = base_config
        else:
            config.update(base_config)

        if robot_name in self.EXTRA_SENSORS:
            config['sensors_obs'] = self.BASE_SENSORS + \
                self.EXTRA_SENSORS[robot_name]
        if robot_name in self.ROBOT_OVERRIDES:
            config.update(self.ROBOT_OVERRIDES[robot_name])

        self.mj_env = Engine(config)
        self.mj_env.reset()
        self._reach = False

        init_low = np.array([1, -1], dtype=np.float32)
        init_high = np.array([2, -2], dtype=np.float32)
        init_space = gym.spaces.Box(low=init_low, high=init_high)

        goal_low = np.array([-2, -2], dtype=np.float32)
        goal_high = np.array([2, 2], dtype=np.float32)
        goal_space = gym.spaces.Box(low=goal_low, high=goal_high)

        init_goal = np.array([0, 0], dtype=np.float32)

        reach_reward = 20

        super(MjEnvBase, self).__init__(self.mj_env.observation_space,
                                        self.mj_env.action_space,
                                        init_space,
                                        goal_space,
                                        init_goal,
                                        reach_reward)

    def update_initial_space(self, low: np.ndarray, high: np.ndarray):
        self.init_space = gym.spaces.Box(low=low, high=high)

    def reset(self, state: np.ndarray = None) -> np.ndarray:
        self._reach = False
        self.mj_env.reset()
        return super(MjEnvBase, self).reset(state)

    def set_goal(self, goal: np.ndarray):
        self.mj_env.set_goal_position(goal_xy=goal[:2])
        return super().set_goal(goal)

    def get_goal(self) -> np.ndarray:
        return self.mj_env.goal_pos[:2]

    def get_obs(self) -> np.ndarray:
        return self.mj_env.obs()

    def reach(self) -> bool:
        return self._reach

    def update_state(self, action: np.ndarray):
        _, _, _, info = self.mj_env.step(action + 0.1)
        self._reach = info.get("goal_met", False)
        info["collision"] = self.collision()

        return info

    def collision(self) -> bool:
        return False

    def get_state(self) -> np.ndarray:
        return self.mj_env.world.robot_pos()[:2]

    def set_state(self, state: np.ndarray):
        # Car reset
        indx = self.mj_env.sim.model.get_joint_qpos_addr("robot")
        sim_state = self.mj_env.sim.get_state()

        sim_state.qpos[indx[0]:indx[0] + 2] = state
        self.mj_env.sim.set_state(sim_state)
        self.mj_env.sim.forward()

    def render(self, mode="human"):
        return self.mj_env.render(mode)

    def close(self):
        self.mj_env.close()

    def _ctrl_reward(self) -> float:
        return self.mj_env.reward(keep_last=True)

    def add_goal_mark(self, pos, color):
        self.mj_env.add_render_callback(lambda: self.mj_env.render_sphere(pos=pos,
                                                                          size=0.3,
                                                                          color=color,
                                                                          alpha=0.5))


class Point(MjEnvBase):
    def __init__(self, config=None):
        super().__init__("point", config, observe_abs_pos=False)

    def set_state(self, state: np.ndarray):
        body_id = self.mj_env.sim.model.body_name2id('robot')
        self.mj_env.sim.model.body_pos[body_id][:2] = state
        self.mj_env.sim.data.body_xpos[body_id][:2] = state
        self.mj_env.sim.forward()


class Car(MjEnvBase):
    def __init__(self, config=None):
        super().__init__("car", config, observe_abs_pos=False)


class Doggo(MjEnvBase):
    def __init__(self, config=None):
        super().__init__("doggo", config, observe_abs_pos=False)


# The goal needs to be controlled for all the environments below
class DronePIDGoalCtrlEnv(DronePIDEnv):
    def __init__(self, gui=False):
        super(DronePIDGoalCtrlEnv, self).__init__(gui=gui)
        # 0-17: pid coef variance
        # 18-21: goal position
        action_space_low = -np.ones(18 + 3, dtype=np.float32)
        action_space_high = np.ones(18 + 3, dtype=np.float32)
        action_space_low[-3:] = self.goal_space.low
        action_space_high[-3:] = self.goal_space.high
        self.action_space = gym.spaces.Box(
            low=action_space_low, high=action_space_high)

    def update_state(self, action: np.ndarray):
        self.set_goal(action[-3:], plot=False)
        super(DronePIDGoalCtrlEnv, self).update_state(action[:-3])

    def get_obs(self):
        return self.robot.get_obs()


class PointGoalCtrlEnv(Point):
    def __init__(self):
        super(Point, self).__init__("point", observe_abs_pos=True)
        # last 2 dimensions are the goal
        action_space_low = np.r_[self.action_space.low, self.goal_space.low]
        action_space_high = np.r_[self.action_space.high, self.goal_space.high]
        self.action_space = Box(action_space_low, action_space_high)

    def update_state(self, action: np.ndarray):
        self.mj_env.set_goal_position(action[-2:])
        super(PointGoalCtrlEnv, self).update_state(action[:-2])


class CarGoalCtrlEnv(Car):
    def __init__(self):
        super(Car, self).__init__("car", observe_abs_pos=True)
        # last 2 dimensions are the goal
        action_space_low = np.r_[self.action_space.low, self.goal_space.low]
        action_space_high = np.r_[self.action_space.high, self.goal_space.high]
        self.action_space = Box(action_space_low, action_space_high)

    def update_state(self, action: np.ndarray):
        self.mj_env.set_goal_position(action[-2:])
        super(CarGoalCtrlEnv, self).update_state(action[:-2])


class DoggoGoalCtrlEnv(Doggo):
    def __init__(self):
        super(Doggo, self).__init__("doggo", observe_abs_pos=True)
        # last 2 dimensions are the goal
        action_space_low = np.r_[self.action_space.low, self.goal_space.low]
        action_space_high = np.r_[self.action_space.high, self.goal_space.high]
        self.action_space = Box(action_space_low, action_space_high)

    def update_state(self, action: np.ndarray):
        self.mj_env.set_goal_position(action[-2:])
        super(DoggoGoalCtrlEnv, self).update_state(action[:-2])
