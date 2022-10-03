import time

import torch

from schrl.common.utils import default_tensor
from schrl.config import DATA_ROOT
from schrl.hrl.policy import GruOdePolicy
from schrl.rebuttal.tasks import get_task


def net_plan(task_name: str, n_trajs: int = 100, max_retry: int = 10):
    task = get_task(task_name)
    pi = GruOdePolicy(2, task.env.path_max_len)
    pi.load_state_dict(torch.load(f"{DATA_ROOT}/rebuttal/planning_time/plan_nets/{task_name}.pth"))

    plan_times = []
    for _ in range(n_trajs):
        start = time.time()
        for _ in range(max_retry):
            init_state = task.env.sample_initial_state(1)
            path = pi.forward(default_tensor(init_state))
            score = task.spec(path[0])

            if score > 0:
                break
        end = time.time()
        plan_times.append(end - start)

    return plan_times
