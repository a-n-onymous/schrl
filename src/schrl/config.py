import logging
from os.path import abspath, dirname

import torch as th

import schrl

# Project and data root
ROOT = dirname(dirname(dirname(abspath(schrl.__file__))))
DATA_ROOT = f"{ROOT}/data"
PROJ_ROOT = f"{ROOT}/src/schrl"

# All tensor will be created as float32 as default
DEFAULT_TENSOR_TYPE = th.float32

# Use CPU as default. Small network on CPU is much faster than on GPU.
DEFAULT_DEVICE = "cpu"  # th.device("cuda") if th.cuda.is_available() else th.device("cpu")
th.set_num_threads(1)

# Log level
LOG_LEVEL = logging.INFO
