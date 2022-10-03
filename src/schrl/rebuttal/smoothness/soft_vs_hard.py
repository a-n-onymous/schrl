import numpy as np
import torch as th
from torch.optim import Adam

from schrl.common.utils import default_tensor, as_numpy
from schrl.hrl.policy import GruOdePolicy
from schrl.rebuttal.tasks import get_task


def train(task_name: str,
          mode: str,
          n_epoch: int = 100,
          n_iter: int = 40,
          batch_trajs: int = 100,
          lr: float = 1e-3,
          stop_acc: float = .8):
    task = get_task(task_name)
    spec = task.spec

    if mode == "soft":
        spec.toggle_soft()
    elif mode == "hard":
        spec.toggle_hard()
    else:
        raise NotImplementedError(f"{mode} is unavailable")

    policy = GruOdePolicy(2, task.env.path_max_len)
    opt = Adam(params=policy.parameters(), lr=lr)
    grad_step = 0

    for _ in range(n_epoch):
        loss_list = []
        score_list = []

        for _ in range(n_iter):
            init_states = task.env.sample_initial_state(batch_trajs)
            paths = policy.forward(default_tensor(init_states))
            scores = th.stack([spec(p) for p in paths])

            if th.sum(scores > 0) / batch_trajs > stop_acc:
                return grad_step

            loss = -th.mean(scores)
            loss_list.append(as_numpy(loss))
            score_list.append(as_numpy(scores).mean())

            opt.zero_grad()
            loss.backward()
            opt.step()
            grad_step += 1

        print(f"{task_name}-{mode}")
        print(f"loss:\t {np.max(loss_list):.2f}|{np.min(loss_list):.2f}|{np.mean(loss_list):.2f}")
        print(f"score:\t {np.max(score_list):.2f}|{np.min(score_list):.2f}|{np.mean(score_list):.2f}")
        print("---")

    return grad_step
