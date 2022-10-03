import random

import numpy as np

from schrl.envs.wrapper import DronePIDEnv, GoalEnv
from schrl.tasks.base import TaskBase
from schrl.tltl.spec import DiffTLTLSpec
from schrl.tltl.template import sequence, coverage, signal, dim_lb, end_state, loop


class DroneSequence(TaskBase):
    def __init__(self, enable_gui: bool = True):
        self.goals = np.array([
            [2, 2, 1],
            [1, -2, 2],
            [-2, -1, 2],
            [0, 2, 1]
        ])
        self.seq_time = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8]
        ]

        super(DroneSequence, self).__init__(8, 2000, enable_gui)

    def _build_env(self, enable_gui) -> GoalEnv:
        env = DronePIDEnv(gui=enable_gui)

        self.duck_size = 20
        env.env_editor.add_duck(pos=np.array([0, 0, 0.099 * self.duck_size * 0.1]), size=self.duck_size)

        for goal in self.goals:
            env.env_editor.add_ball(goal, color=(0, 0, 1, 0.5))
        return env

    def _build_spec(self) -> DiffTLTLSpec:
        goal_spec = sequence(self.goals, close_radius=0.5, seq_time=self.seq_time)
        ground_spec = dim_lb(0.1, dim=2)

        spec = goal_spec & ground_spec
        return spec

    def demo(self, n_trajs: int) -> np.ndarray:
        dataset = []
        for _ in range(n_trajs):
            traj = [[self.env.sample_init_state()]]
            for i in range(len(self.goals) - 1):
                part_traj = np.linspace(self.goals[i], self.goals[i + 1], endpoint=False, num=2)
                traj.append(part_traj)
            traj.append([self.goals[-1]])
            dataset.append(np.concatenate(traj))

        return np.array(dataset)


class DroneBranch(TaskBase):
    def __init__(self, enable_gui: bool = True):
        self.branch_1 = np.array([
            [1, 0, 1],
            [-1, 0, 1],
        ])
        self.branch_2 = np.array([
            [1, -1, 1],
            [-1, 1, 1],
        ])
        self.branch_3 = np.array([
            [0, 1, 2],
            [0, -1, 2],
        ])

        self.seq_time = [
            [1, 2],
            [3, 4]
        ]

        super(DroneBranch, self).__init__(4, 1000, enable_gui)

    def _build_env(self, enable_gui: bool):
        env = DronePIDEnv(enable_gui)

        for goal in self.branch_1:
            env.env_editor.add_ball(goal, color=(1, 0, 0, 0.5))

        for goal in self.branch_2:
            env.env_editor.add_ball(goal, color=(0, 1, 1, 0.5))

        for goal in self.branch_3:
            env.env_editor.add_ball(goal, color=(0, 0, 1, 0.5))

        return env

    def _build_spec(self) -> DiffTLTLSpec:
        b1 = sequence(self.branch_1, seq_time=self.seq_time, name_prefix="b1")
        b2 = sequence(self.branch_2, seq_time=self.seq_time, name_prefix="b2")
        b3 = sequence(self.branch_3, seq_time=self.seq_time, name_prefix="b3")

        final_1 = end_state(self.branch_1[-1])
        final_2 = end_state(self.branch_2[-1])
        final_3 = end_state(self.branch_3[-1])

        imply_final_1 = b1.implies(final_1)
        imply_final_2 = b2.implies(final_2)
        imply_final_3 = b3.implies(final_3)

        g = dim_lb(0.1, dim=2)

        return g & imply_final_1 & imply_final_2 & imply_final_3 & (b1 | b2 | b3)

    def demo(self, n_trajs: int) -> np.ndarray:
        dataset = []
        for _ in range(n_trajs):
            traj = [[self.env.sample_init_state()]]
            branch = random.choice([self.branch_1, self.branch_2, self.branch_3])
            for i in range(len(branch) - 1):
                part_traj = np.linspace(branch[i], branch[i + 1], endpoint=False, num=2)
                traj.append(part_traj)
            traj.append([branch[-1]])
            dataset.append(np.concatenate(traj))

        return np.array(dataset)


class DroneCover(TaskBase):

    def __init__(self, enable_gui: bool = True):
        self.goals = np.array([
            [2, 2, 1],
            [1, -2, 2],
            [-2, -1, 2],
            [0, 2, 1]
        ])[::-1]

        super(DroneCover, self).__init__(8, 2000, enable_gui)

    def _build_env(self, enable_gui: bool):
        env = DronePIDEnv(enable_gui)

        self.duck_size = 15
        env.env_editor.add_duck(pos=np.array([0, 0, 0.099 * self.duck_size * 0.1]), size=self.duck_size)

        for goal in self.goals:
            env.env_editor.add_ball(goal, color=(0, 0, 1, 0.5))
        return env

    def _build_spec(self) -> DiffTLTLSpec:
        coverage_spec = coverage(self.goals, close_radius=0.5)
        ground_spec = dim_lb(0.1, dim=2)

        spec = coverage_spec & ground_spec

        return spec

    def demo(self, n_trajs: int) -> np.ndarray:
        dataset = []
        for _ in range(n_trajs):
            traj = [[self.env.sample_init_state()]]
            for i in range(len(self.goals) - 1):
                part_traj = np.linspace(self.goals[i], self.goals[i + 1], endpoint=False, num=2)
                traj.append(part_traj)
            traj.append([self.goals[-1]])
            dataset.append(np.concatenate(traj))

        return np.array(dataset)


class DroneLoop(TaskBase):

    def __init__(self, enable_gui: bool = False):
        self.waypoints = ([-1, -1, 1],
                          [1, 1, 2],
                          [2, 1, 1])
        super(DroneLoop, self).__init__(21, 4000, enable_gui)

    def _build_env(self, enable_gui: bool):
        env = DronePIDEnv(enable_gui)
        for goal in self.waypoints:
            env.env_editor.add_ball(goal, color=(0, 0, 1, 0.5))
        return env

    def _build_spec(self) -> DiffTLTLSpec:
        spec = loop(self.waypoints, close_radius=0.5)
        g = dim_lb(0.1, dim=2)

        return spec & g

    def demo(self, n_trajs: int) -> np.ndarray:
        dataset = []
        for _ in range(n_trajs):
            traj = [self.env.sample_init_state()]
            traj_len_so_far = 0
            not_yet = True

            while not_yet:
                for wp in self.waypoints:
                    traj.append(wp)
                    traj_len_so_far += 1
                    if traj_len_so_far >= self.path_max_len:
                        not_yet = False
                        break

            dataset.append(traj)

        return np.array(dataset)


class DroneSignal(TaskBase):
    def __init__(self, enable_gui: bool):
        self.loop_waypoints = ([-1, -1, 1.5], [1, 1, 2], [1, 1, 1])
        self.final_goal = [-2, -2, 2]
        self.until_time = 10

        super(DroneSignal, self).__init__(self.until_time + 2, 4000, enable_gui)

    def _build_env(self, enable_gui: bool):
        env = DronePIDEnv(enable_gui)

        for i, goal in enumerate(self.loop_waypoints):
            env.env_editor.add_ball(goal, color=(1, 0, i / len(goal), 0.5))

        env.env_editor.add_ball(self.final_goal, color=(1, 1, 0, 0.5))

        return env

    def _build_spec(self) -> DiffTLTLSpec:
        spec = signal(self.loop_waypoints, self.final_goal, self.until_time, close_radius=0.5)
        g = dim_lb(0.1, dim=2)

        return spec & g

    def demo(self, n_trajs: int) -> np.ndarray:
        dateset = []

        for _ in range(n_trajs):
            traj = [self.env.sample_init_state()]
            i = 0
            while i < self.until_time:
                traj.append(self.loop_waypoints[i % len(self.loop_waypoints)])
                i += 1
            traj.append(self.final_goal)
            dateset.append(traj)

        return np.array(dateset)
