from typing import Union

import numpy as np
import torch as th

from schrl.config import DEFAULT_DEVICE, DEFAULT_TENSOR_TYPE


class Path:
    def __init__(self,
                 traj_array: Union[np.ndarray, th.Tensor]):
        if isinstance(traj_array, th.Tensor):
            self.traj_array: np.ndarray = traj_array.detach().cpu().numpy()
        else:
            self.traj_array: np.ndarray = traj_array

    @classmethod
    def _build(cls, traj_array):
        return cls(traj_array)

    def to_torch(self,
                 t: int = None,
                 dtype=DEFAULT_TENSOR_TYPE,
                 device=DEFAULT_DEVICE) -> th.Tensor:
        if t is None:
            return th.tensor(self.traj_array, dtype=dtype, device=device)
        else:
            return th.tensor(self.traj_array[t:t + 1], dtype=dtype, device=device)

    def insert(self, indx, state: Union[np.ndarray, th.Tensor]):
        if isinstance(state, th.Tensor):
            state: np.ndarray = state.detach().cpu().numpy()
        self.traj_array = np.insert(self.traj_array, [indx], state, axis=0)

    def __getitem__(self, item) -> 'Path':
        return self._build(self.traj_array[item])

    def __iter__(self):
        return self.traj_array.__iter__()

    def __len__(self):
        return self.traj_array.__len__()

    def __repr__(self):
        repr_str = ""
        for state in self.traj_array:
            repr_str += f"{state} "
        return repr_str
