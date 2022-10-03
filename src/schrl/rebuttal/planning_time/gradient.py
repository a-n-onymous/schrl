import time

import numpy as np
import torch as th
from torch.nn import Parameter
from torch.optim import Adam

from schrl.common.utils import default_tensor
from schrl.rebuttal.tasks import get_task


def grad_plan(task_name: str, n_trajs: int = 100, max_grad_step: int = 4000, max_retry: int = 5, lr=1e-3):
    task = get_task(task_name)

    path_parameter = Parameter(th.zeros(task.env.path_max_len, 2))
    opt = Adam(params=[path_parameter], lr=lr)

    plan_times = []
    for _ in range(n_trajs):
        start = time.time()
        for _ in range(max_retry):
            path = np.array([task.env.sample_initial_state(1)[0] for _ in range(task.env.path_max_len)])
            path = default_tensor(path)
            path.requires_grad = True
            path_parameter.data = path.data
            success = False

            for _ in range(max_grad_step):
                score = task.spec(path_parameter)
                loss = -score

                opt.zero_grad()
                loss.backward()
                opt.step()

                if score > 0:
                    success = True
                    break

            if success:
                break

        end = time.time()
        plan_times.append(end - start)

    return plan_times
