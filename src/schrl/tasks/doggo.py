import random

import numpy as np

from schrl.envs.wrapper import GoalEnv, Doggo
from schrl.tasks.base import TaskBase
from schrl.tltl.spec import DiffTLTLSpec
from schrl.tltl.template import sequence, ith_state, loop, signal, coverage


class DoggoSequence(TaskBase):
    def __init__(self, enable_gui: bool = True):
        self.goals = np.array([
            [2, 2],
            [-1, 2],
            [-2, -1],
            [1, -2]
        ])
        self.seq_time = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8]
        ]

        super(DoggoSequence, self).__init__(8, 1000, enable_gui)

    def _build_env(self, enable_gui=True) -> GoalEnv:
        config = {
            'walls_num': 17,
            'walls_locations': [
                [0, 0],
                [0, 0.5], [0, 1], [0, 3], [0, 3.5],
                [0.5, 0], [1, 0], [3, 0], [3.5, 0],
                [0, -0.5], [0, -1], [0, -3], [0, -3.5],
                [-0.5, 0], [-1, 0], [-3, 0], [-3.5, 0],
            ],
            'walls_size': 0.25,
        }
        env = Doggo(config)
        for pos in self.goals:
            env.add_goal_mark(pos, color=(0, 1, 1, 0.5))
        return env

    def _build_spec(self) -> DiffTLTLSpec:
        goal_spec = sequence(self.goals, close_radius=0.3, seq_time=self.seq_time)
        return goal_spec

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


class DoggoCover(TaskBase):
    def __init__(self, enable_gui: bool = True):
        self.goals = np.array([
            [2, 2],
            [1, -2],
            [-2, -1],
            [-1, 2]
        ])

        super(DoggoCover, self).__init__(8, 1000, enable_gui)

    def _build_env(self, enable_gui=True) -> GoalEnv:
        config = {
            'walls_num': 17,
            'walls_locations': [
                [0, 0],
                [0, 0.5], [0, 1], [0, 3], [0, 3.5],
                [0.5, 0], [1, 0], [3, 0], [3.5, 0],
                [0, -0.5], [0, -1], [0, -3], [0, -3.5],
                [-0.5, 0], [-1, 0], [-3, 0], [-3.5, 0],
            ],
            'walls_size': 0.25,
        }
        env = Doggo(config)
        for pos in self.goals:
            env.add_goal_mark(pos, color=(0, 1, 1, 0.5))
        return env

    def _build_spec(self) -> DiffTLTLSpec:
        goal_spec = coverage(self.goals, close_radius=0.5)
        return goal_spec

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


class DoggoBranch(TaskBase):
    def __init__(self, enable_gui: bool = True):
        self.branches = np.array([
            [[0, 2], [0, -2]],
            [[2, -2], [-2, 2]],
            [[2, 0], [-2, 0]]
        ])

        self.seq_time = [
            [1, 2],
            [3, 4]
        ]

        super(DoggoBranch, self).__init__(4, 1000, enable_gui)

    def _build_env(self, enable_gui: bool):
        config = {
            'walls_num': 17,
            'walls_locations': [
                [0, 0],
                [0, 0.5], [0, 1], [0, 3], [0, 3.5],
                [0.5, 0], [1, 0], [3, 0], [3.5, 0],
                [0, -0.5], [0, -1], [0, -3], [0, -3.5],
                [-0.5, 0], [-1, 0], [-3, 0], [-3.5, 0],
            ],
            'walls_size': 0.25,
        }
        env = Doggo(config)

        init_low = np.array([-2, -2], dtype=np.float32)
        init_high = np.array([-1, -1], dtype=np.float32)
        env.update_initial_space(init_low, init_high)

        for goal in self.branches[0]:
            env.add_goal_mark(goal, color=(1, 0, 0, 0.5))

        for goal in self.branches[1]:
            env.add_goal_mark(goal, color=(0, 1, 0, 0.5))

        for goal in self.branches[2]:
            env.add_goal_mark(goal, color=(0, 0, 1, 0.5))

        return env

    def _build_spec(self) -> DiffTLTLSpec:
        start_1 = ith_state((self.branches[0][0]), i=1, close_radius=0.6)
        start_2 = ith_state((self.branches[1][0]), i=1, close_radius=0.6)
        start_3 = ith_state((self.branches[2][0]), i=1, close_radius=0.6)

        b1 = sequence(self.branches[0], close_radius=0.6, seq_time=self.seq_time, name_prefix="b1")
        b2 = sequence(self.branches[1], close_radius=0.6, seq_time=self.seq_time, name_prefix="b2")
        b3 = sequence(self.branches[2], close_radius=0.6, seq_time=self.seq_time, name_prefix="b3")

        imply_final_1 = start_1.implies(b1)
        imply_final_2 = start_2.implies(b2)
        imply_final_3 = start_3.implies(b3)

        return imply_final_1 & imply_final_2 & imply_final_3 & (start_1 | start_2 | start_3)

    def demo(self, n_trajs: int) -> np.ndarray:
        dataset = []
        mid_points = np.array([[0, 2.5], [2.5, 2.5], [0, 2.5]])
        for _ in range(n_trajs):
            traj = [[self.env.sample_init_state()]]
            indx = random.choice([1])
            for j in range(len(self.branches[indx]) - 1):
                part_traj = np.array([self.branches[indx][j], mid_points[indx]])
                traj.append(part_traj)
            traj.append([self.branches[indx][-1]])
            dataset.append(np.concatenate(traj))

        return np.array(dataset)


class DoggoLoop(TaskBase):
    def __init__(self, enable_gui: bool = False):
        self.waypoints = ([-2, -2],
                          [-2, 2],
                          [2, 2],
                          [2, -2])
        super(DoggoLoop, self).__init__(17, 2000, enable_gui)

    def _build_env(self, enable_gui: bool):
        config = {
            'walls_num': 17,
            'walls_locations': [
                [0, 0],
                [0, 0.5], [0, 1], [0, 3], [0, 3.5],
                [0.5, 0], [1, 0], [3, 0], [3.5, 0],
                [0, -0.5], [0, -1], [0, -3], [0, -3.5],
                [-0.5, 0], [-1, 0], [-3, 0], [-3.5, 0],
            ],
            'walls_size': 0.25,
        }
        env = Doggo(config)
        for goal in self.waypoints:
            env.add_goal_mark(goal, color=(1, 0, 0, 0.5))
        return env

    def _build_spec(self) -> DiffTLTLSpec:
        spec = loop(self.waypoints, close_radius=0.5)

        return spec

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


class DoggoSignal(TaskBase):
    def __init__(self, enable_gui: bool):
        self.loop_waypoints = ([-2, -2],
                               [-2, 2],
                               [2, 2],
                               [2, -2])
        self.final_goal = [0, 0]
        self.until_time = 10

        super(DoggoSignal, self).__init__(self.until_time + 2, 2000, enable_gui)

    def _build_env(self, enable_gui: bool):
        env = Doggo()

        for i, goal in enumerate(self.loop_waypoints):
            env.add_goal_mark(goal, color=(0, 1, i / len(goal), 0.5))

        env.add_goal_mark(self.final_goal, color=(1, 1, 0, 0.5))

        return env

    def _build_spec(self) -> DiffTLTLSpec:
        spec = signal(self.loop_waypoints, self.final_goal, self.until_time, close_radius=0.8)

        return spec

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
