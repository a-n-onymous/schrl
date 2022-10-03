from typing import Tuple

import numpy as np
import pandas as pd
import torch as th
from pandas import DataFrame

from schrl.common.utils import as_numpy, default_tensor
from schrl.hrl.policy import GruOdePolicy
from schrl.rebuttal.tasks import TaskBase


class SpecConstrainedPG:
    def __init__(self,
                 task: TaskBase,
                 constrained: bool = True,
                 # rollout parameters
                 n_trajs: int = 100,
                 # policy parameters
                 policy_cls=GruOdePolicy,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 der_bound: float = 10.0,
                 # optimization parameters
                 optimizer_cls=th.optim.RMSprop,
                 lr: float = 1e-3,
                 update_epochs: int = 100,
                 # experimental
                 disable_pg: bool = False
                 ):
        self.task = task
        self.env = task.env
        self.spec = task.spec
        self.n_trajs = n_trajs
        self.constrained = constrained
        self.disable_pg = disable_pg

        self.policy = policy_cls(
            self.env.input_size, self.env.path_max_len, hidden_size, num_layers, der_bound)
        self.optimizer = optimizer_cls(params=self.policy.parameters(), lr=lr)
        self.update_epochs = update_epochs

        self.trajs = None
        self.init_state_tensor = default_tensor(0)

    def learn(self, total_rollout_steps: int):
        cur_n_traj = 0
        while cur_n_traj < total_rollout_steps:
            # collect data
            self.rollout()

            # update with constrained policy gradient
            opt_info, sat_ratio = self.constrained_pg_update()

            # track total training step
            cur_n_traj += self.n_trajs

            # log
            self._log(opt_info, sat_ratio)

            if sat_ratio > 0.95:
                # 95% trajectory has score > 0
                break

        return cur_n_traj

    def rollout(self):
        self.init_state_tensor = default_tensor(
            self.env.sample_initial_state(self.n_trajs))
        # no step for only planning env

    def constrained_pg_update(self) -> Tuple[DataFrame, float]:
        pg_losses = []
        cnstr_losses = []
        spec_scores_list = []
        last_spec_scores = default_tensor(0)
        lam = th.zeros(self.n_trajs)

        for _ in range(self.update_epochs):
            self.trajs = self.policy.forward(self.init_state_tensor)
            spec_scores = th.stack([self.spec(traj) for traj in self.trajs])
            spec_scores_list.append(as_numpy(spec_scores))

            loss = default_tensor(0)
            if not self.disable_pg:
                log_probs = self.policy.log_prob(self.trajs)

                score_min = th.min(spec_scores)
                advantage = spec_scores.detach() - score_min.detach()
                pg_loss = th.mean(-log_probs * advantage)
                pg_losses.append(as_numpy(pg_loss))
                loss += pg_loss

            if self.constrained:
                lam += spec_scores.detach()
                lam = th.clip(lam, -th.inf, -1.0)
                cnstr_loss = th.mean(lam * spec_scores)
                cnstr_losses.append(as_numpy(cnstr_loss))
                loss += cnstr_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            last_spec_scores = spec_scores

        last_spec_scores = as_numpy(last_spec_scores)
        stat_dict = {
            "spec_scores": [np.mean(spec_scores_list), np.max(spec_scores_list), np.min(spec_scores_list)],
            "last_spec_scores": [np.mean(last_spec_scores), np.max(last_spec_scores), np.min(last_spec_scores)]
        }
        if self.constrained:
            stat_dict["cnstr_loss"] = [np.mean(cnstr_losses), np.max(
                cnstr_losses), np.min(cnstr_losses)]
        if not self.disable_pg:
            stat_dict["pg_loss"] = [
                np.mean(pg_losses), np.max(pg_losses), np.min(pg_losses)]

        stat = pd.DataFrame.from_dict(
            stat_dict, columns=["mean", "max", "min"], orient="index")

        sat_ratio = np.sum(last_spec_scores > 0) / len(last_spec_scores)

        return stat, float(sat_ratio)

    def _log(self, opt_info_df: pd.DataFrame, sat_ratio: float):
        print(opt_info_df)
        print("sat ratio: ", sat_ratio)
