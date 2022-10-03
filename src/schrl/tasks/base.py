from abc import ABC, abstractmethod

import numpy as np

from schrl.envs.wrapper import GoalEnv
from schrl.tltl.spec import DiffTLTLSpec


class TaskBase(ABC):
    def __init__(self, path_max_len: int, time_limit: int, enable_gui: bool):
        self.path_max_len = path_max_len
        self.time_limit = time_limit
        self.env = self._build_env(enable_gui)
        self.spec = self._build_spec()
        self.enable_gui = enable_gui

    @abstractmethod
    def _build_env(self, enable_gui: bool) -> GoalEnv:
        raise NotImplementedError()

    @abstractmethod
    def _build_spec(self) -> DiffTLTLSpec:
        raise NotImplementedError()

    @abstractmethod
    def demo(self, n_trajs: int) -> np.ndarray:
        raise NotImplementedError()

    def __repr__(self):
        return repr(self.spec)
