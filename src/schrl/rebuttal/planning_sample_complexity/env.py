import numpy as np


class OnlyPlanningEnv:
    def __init__(self,
                 input_size: int,
                 path_max_len: int,
                 init_low: np.ndarray,
                 init_high: np.ndarray):
        self.input_size = input_size
        self.path_max_len = path_max_len
        self.init_low = init_low
        self.init_high = init_high

    def sample_initial_state(self, n_states: int):
        return np.random.uniform(self.init_low, self.init_high, size=(n_states,) + self.init_low.shape)
