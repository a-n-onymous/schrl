from abc import ABC, abstractmethod
from typing import Callable

import torch as th
from termcolor import colored
from torch import nn


class PredicateBase(ABC):
    """
    Predicate for a single state or a batch of states.
    One state is mapped to one value.
    If Output is greater than 0 means that the state satisfies the predicate;
    If output is smaller than 0 means that the state does not satisfy the predicate.
    """

    def __init__(self, name: str, color: str = None):
        self.name = name
        self.color = color

    @abstractmethod
    def forward(self, state: th.Tensor) -> th.Tensor:
        pass

    def __call__(self, state: th.Tensor) -> th.Tensor:
        assert 0 < len(state.shape) <= 2, "Predicate only takes one state or batch of states as input"

        pred_value = self.forward(state)

        return pred_value

    def __repr__(self):
        return colored(self.name, self.color)


class ProgrammablePredicate(PredicateBase):
    """
    Programmable Predicate
    """

    def __init__(self, diff_func: Callable[[th.Tensor], th.Tensor], name: str):
        """
        Init
        :param diff_func: differentiable function (must support batch input)
        :param name: predicate name
        """
        super(ProgrammablePredicate, self).__init__(name, "blue")
        self.diff_func = diff_func

    def forward(self, state: th.Tensor) -> th.Tensor:
        return self.diff_func(state)


class NeuralPredicate(PredicateBase):
    """
    Neural Network Predicate
    """

    def __init__(self, model: nn.Module, name: str):
        super(NeuralPredicate, self).__init__(name, "green")
        self.model = model

    def forward(self, state: th.Tensor) -> th.Tensor:
        return self.model(state)
