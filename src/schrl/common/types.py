from typing import NewType, Union, Tuple, List

import numpy as np

Vec = NewType("Vec", Union[List[float], Tuple[float], np.ndarray])
