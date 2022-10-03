import os
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import CallbackList

from schrl.config import DEFAULT_DEVICE
from schrl.hrl.buffer import PlanLayerBuffer
from schrl.hrl.policy import GruOdePolicy
from schrl.hrl.sb3_callback import Dispatcher, CtrlRewardCollector, Updater, Render
from schrl.tasks import TaskBase


class PlanLayerBase(ABC):
    def __init__(self, task: TaskBase):
        self.task = task
        self.env = task.env
        self.spec = task.spec
        self.path_max_len = task.path_max_len
        self.enable_gui = self.task.enable_gui

    @property
    @abstractmethod
    def callback_list(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def disable_learning(self):
        pass

    @abstractmethod
    def enable_learning(self):
        pass

    @abstractmethod
    def predict(self, init_state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, task: TaskBase, device: str = DEFAULT_DEVICE) -> 'PlanLayerBase':
        pass


class LearnablePlanLayer(PlanLayerBase):
    def __init__(self,
                 task: TaskBase,
                 policy_kwargs: Dict = None,
                 opt_kwargs: Dict = None,
                 update_intv: int = 8,
                 n_epoch: int = 1000,
                 batch_size: int = 8,
                 perf_coef: float = 0.1,
                 clip_range: float = 0.05,
                 demo_dataset: np.ndarray = None,
                 device: str = DEFAULT_DEVICE):
        # unwrapped environment must be GoalEnv
        super(LearnablePlanLayer, self).__init__(task)

        # planning policy
        self.policy_kwargs = policy_kwargs
        if policy_kwargs is None:
            self.policy_kwargs = {}
        self.policy = GruOdePolicy(self.env.goal_space_size,
                                   self.path_max_len, **self.policy_kwargs)
        self.policy.to(device)

        # on-policy buffer
        self.buffer = PlanLayerBuffer()

        # render callbacks
        self.render = Render()
        # callback for collecting control rewards
        self.ctrl_reward_collector = CtrlRewardCollector(self.buffer)
        # callback for checking reaching sub-goals and set new subgoal
        self.dispatcher = Dispatcher(self.policy, self.buffer)
        # callback for constrained policy gradient update
        self.updater = Updater(self.policy,
                               self.buffer,
                               self.spec,
                               update_intv,
                               n_epoch,
                               batch_size,
                               perf_coef,
                               clip_range,
                               demo_dataset,
                               opt_kwargs)

        # callbacks for training and rendering
        if self.enable_gui:
            self._callback_list = CallbackList(
                [self.render, self.dispatcher, self.ctrl_reward_collector, self.updater])
        else:
            self._callback_list = CallbackList(
                [self.dispatcher, self.ctrl_reward_collector, self.updater])

    def enable_learning(self):
        if self.callback_list.callbacks[-1] is not self.updater:
            self._callback_list.callbacks.extend([self.ctrl_reward_collector, self.updater])

    def disable_learning(self):
        while self.callback_list.callbacks[-1] is not self.dispatcher:
            self._callback_list.callbacks.pop()

    @property
    def callback_list(self):
        return self._callback_list

    def predict(self, init_state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.policy.predict(init_state, deterministic)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        th.save(self.policy.state_dict(), f"{path}/plan_net.pth")

    @classmethod
    def load(cls, model_path: str, task: TaskBase, device: str = DEFAULT_DEVICE) -> 'PlanLayerBase':
        layer = cls(task)
        plan_net_state_dict = th.load(f"{model_path}/plan_net.pth", map_location=device)
        layer.policy.load_state_dict(plan_net_state_dict)

        return layer
