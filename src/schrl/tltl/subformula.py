from abc import ABC, abstractmethod
from typing import Callable

import torch as th
from termcolor import colored
from torch import nn


class SubFormulaBase(ABC):
    """
    Sub-formular for a single trajectory or a batch of trajectory.
    One trajectory to one value.
    Greater than 0 means that the state satisfies the sub-formular;
    Smaller than 0 means that the state does not satisfy the sub-formular,.
    """

    def __init__(self,
                 name: str,
                 start: int = 0,
                 end: int = None,
                 color: str = None):
        self.name = name
        self.start = start
        self.end = end
        self.color = color

    @abstractmethod
    def forward(self, traj: th.Tensor) -> th.Tensor:
        pass

    def __call__(self, traj: th.Tensor) -> th.Tensor:
        assert 2 <= len(traj.shape) <= 3, "Sub-formula only takes one trajectory" \
                                          " or batch of trajectories as input"

        if len(traj.shape) == 2:
            forward_traj = traj[self.start:self.end]
            pred_value = self.forward(forward_traj)
            assert len(pred_value.shape) == 0, "Check forward implementation to align input and output"
        else:  # len(traj.shape) == 3
            forward_traj = traj[:, self.start, self.end]
            pred_value = self.forward(forward_traj)
            assert len(pred_value) == len(traj), "Check forward implementation to align input and output"

        return pred_value

    def __repr__(self):
        return colored(self.name, self.color)


class ProgrammableSubFormula(SubFormulaBase):
    """
    Programmable Sub-Formula
    """

    def __init__(self,
                 diff_func: Callable[[th.Tensor], th.Tensor],
                 name: str,
                 start: int = 0,
                 end: int = None):
        super(ProgrammableSubFormula, self).__init__(name, start, end, "blue")
        self.diff_func = diff_func

    def forward(self, traj: th.Tensor) -> th.Tensor:
        return self.diff_func(traj)


class NeuralSubFormula(SubFormulaBase):
    """
    Neural Network Sub-Formula
    It is not used in this project
    """

    def __init__(self,
                 model: nn.Module, name: str,
                 start: int = 0,
                 end: int = None):
        super(NeuralSubFormula, self).__init__(name, start, end, "green")
        self.model = model

    def forward(self, traj: th.Tensor) -> th.Tensor:
        return self.model(traj)
