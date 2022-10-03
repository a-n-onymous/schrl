from abc import ABC, abstractmethod

import numpy as np

from schrl.rebuttal.planning_sample_complexity.env import OnlyPlanningEnv
from schrl.tltl.spec import DiffTLTLSpec
from schrl.tltl.template import branch, coverage, loop, sequence, signal


class TaskBase(ABC):
    def __init__(self):
        self.spec = self._build_spec()
        self.env = self._build_env()

    @abstractmethod
    def _build_spec(self) -> DiffTLTLSpec:
        pass

    @abstractmethod
    def _build_env(self) -> OnlyPlanningEnv:
        pass

    def __repr__(self):
        return repr(self.spec)


class Seq(TaskBase):
    def _build_spec(self) -> DiffTLTLSpec:
        self.waypoints = [
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]
        self.seq_times = [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5]
        ]
        self.close_r = 0.3

        return sequence(self.waypoints, self.seq_times, self.close_r)

    def _build_env(self) -> OnlyPlanningEnv:
        return OnlyPlanningEnv(input_size=2,
                               path_max_len=5,
                               init_low=np.array([0.0, 0.0]),
                               init_high=np.array([0.5, 0.5]))


class Cover(TaskBase):
    def _build_spec(self) -> DiffTLTLSpec:
        self.waypoints = [
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]
        self.close_r = 0.3

        return coverage(self.waypoints, self.close_r)

    def _build_env(self) -> OnlyPlanningEnv:
        return OnlyPlanningEnv(input_size=2,
                               path_max_len=5,
                               init_low=np.array([0.0, 0.0]),
                               init_high=np.array([0.5, 0.5]))


class Branch(TaskBase):

    def _build_spec(self) -> DiffTLTLSpec:
        self.branches = [
            [[1, 1], [-1, 1]],
            [[-1, -1], [1, -1]]
        ]

        self.close_r = 0.3
        return branch(self.branches, self.close_r)

    def _build_env(self) -> OnlyPlanningEnv:
        return OnlyPlanningEnv(input_size=2,
                               path_max_len=3,
                               init_low=np.array([0.0, 0.0]),
                               init_high=np.array([0.5, 0.5]))


class Loop(TaskBase):

    def _build_spec(self) -> DiffTLTLSpec:
        self.loop_wps = [[1, 1], [1, -1]]
        self.close_r = 0.5

        return loop(self.loop_wps, self.close_r)

    def _build_env(self) -> OnlyPlanningEnv:
        return OnlyPlanningEnv(input_size=2,
                               path_max_len=16,
                               init_low=np.array([0.0, 0.0]),
                               init_high=np.array([0.5, 0.5]))


class Signal(TaskBase):

    def _build_spec(self) -> DiffTLTLSpec:
        self.loop_wps = [[1, 1], [1, -1]]
        self.final_goal = [-1, -1]
        self.until_time = 4
        self.close_r = 0.5

        return signal(self.loop_wps, self.final_goal, self.until_time, self.close_r)

    def _build_env(self) -> OnlyPlanningEnv:
        return OnlyPlanningEnv(input_size=2,
                               path_max_len=6,
                               init_low=np.array([0.0, 0.0]),
                               init_high=np.array([0.5, 0.5]))


def get_task(task_name: str):
    if task_name == "seq":
        return Seq()
    elif task_name == "cover":
        return Cover()
    elif task_name == "branch":
        return Branch()
    elif task_name == "loop":
        return Loop()
    elif task_name == "signal":
        return Signal()
    else:
        raise NotImplementedError(f"{task_name} is not implemented")
