from typing import Tuple

import torch as th
from torch import nn


class MLP(nn.Module):
    def __init__(self,
                 inpt_dim: int,
                 out_dim: int,
                 hidden_sizes: Tuple = (128, 128),
                 squash_output: bool = False):
        super(MLP, self).__init__()
        relu = th.nn.ReLU()
        tanh = th.nn.Tanh()
        self.squash_output = squash_output

        assert len(hidden_sizes) > 0
        layers = [th.nn.Linear(inpt_dim, hidden_sizes[0])]
        for i in range(len(hidden_sizes) - 1):
            layers.append(relu)
            layers.append(th.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(relu)
        layers.append(th.nn.Linear(hidden_sizes[-1], out_dim))
        if self.squash_output:
            layers.append(tanh)

        self.mlp = th.nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.mlp(x)


class GRUEulerNeuralOde(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 der_bound: float = 10.0):
        super(GRUEulerNeuralOde, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.linear_layer = nn.Linear(hidden_size, input_size)
        self.tanh = nn.Tanh()
        self.der_bound = der_bound

    def forward(self, x: th.Tensor, hn: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if hn is not None:
            hn = hn.detach()

        logist, hn = self.gru(x, hn)
        der = self.linear_layer(logist)
        der = self.tanh(der)

        return x + der * self.der_bound, logist, hn
