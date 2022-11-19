from typing import Dict

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn.functional import mse_loss

from schrl.common.utils import default_tensor, as_numpy
from schrl.config import LOG_LEVEL
from schrl.envs import GoalEnv
from schrl.hrl.buffer import PlanLayerBuffer
from schrl.hrl.policy import GruOdePolicy
from schrl.tltl.spec import DiffTLTLSpec


class Render(BaseCallback):
    def _on_step(self) -> bool:
        if self.training_env.num_envs > 1:
            raise NotImplementedError("Do not support vec env for now")

        env: GoalEnv = self.training_env.envs[0]  # noqa
        env.render()

        return True


class CtrlRewardCollector(BaseCallback):
    def __init__(self, buffer: PlanLayerBuffer):
        super(CtrlRewardCollector, self).__init__()
        self.buffer = buffer
        self.step_to_reach = 0.0
        self.prev_goal = None
        self.cum_reward = 0.0

    def _on_step(self) -> bool:
        if self.training_env.num_envs > 1:
            raise NotImplementedError("Do not support vec env for now")

        done = self.locals["dones"][0]
        reward = self.locals["rewards"][0]

        self.cum_reward += reward

        if done:
            self.buffer.add_ctrl_reward(self.cum_reward)
            self.cum_reward = 0.0

        return True


class Dispatcher(BaseCallback):
    def __init__(self,
                 plan_policy: GruOdePolicy,
                 buffer: PlanLayerBuffer):
        super(Dispatcher, self).__init__()

        self.policy = plan_policy
        self.buffer = buffer

        self.planned_path = []
        self.goal_indx = 0
        self.step_nums = 0
        self.ep_step_to_reach = []
        self.path_scores = []

    def _on_training_start(self) -> None:
        self.logger.set_level(LOG_LEVEL)

    def _on_step(self) -> bool:
        if self.training_env.num_envs > 1:
            raise NotImplementedError("Do not support vec env for now. "
                                      "TODO: It can be implemented with a for loop")

        env: GoalEnv = self.training_env.envs[0]  # noqa
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]
        self.step_nums += 1

        # replan when planned path is empty
        if len(self.planned_path) == 0:
            planned_path_tensor = self.policy.forward(default_tensor(env.init_state))[0]
            self.planned_path = as_numpy(planned_path_tensor)

            # add planner policy predict results to buffer
            self.buffer.add_traj(self.planned_path)
            self.buffer.add_log_prob(self.policy.log_prob(planned_path_tensor).detach())

            # clean counters
            self.goal_indx = 0
            self.step_nums = 0

            # log
            self.logger.debug("replan")
            self.logger.debug(f"initial observation: {self.locals['new_obs']}")

        # reach one sub-goal
        if (self.goal_indx == 0 or info["reach"]) and self.goal_indx < len(self.planned_path):
            goal = self.planned_path[self.goal_indx]
            self.logger.debug("set new goal: ", goal)
            env.set_goal(goal)
            self.goal_indx += 1

        # reach all goals - task complete
        if self.goal_indx == len(self.planned_path) and self.step_nums >= 0:
            if not done:
                self.logger.info(f"reach step: {self.step_nums}")
                self.ep_step_to_reach.append(self.step_nums)
                env.stop_in_next_step()
            else:
                self.planned_path = []

        # fail to reach all goals in limit steps
        if done and self.goal_indx < len(self.planned_path):
            self.logger.info(f"fail to reach in {self.step_nums} steps")
            self.planned_path = []

        return True

    def on_rollout_end(self) -> None:
        if self.ep_step_to_reach:
            step_to_reach_mean = np.mean(self.ep_step_to_reach)
        else:
            step_to_reach_mean = np.inf
        self.logger.record("rollout/step_to_reach_mean", step_to_reach_mean)
        self.ep_step_to_reach = []


class Updater(BaseCallback):
    def __init__(self,
                 policy: GruOdePolicy,
                 buffer: PlanLayerBuffer,
                 spec: DiffTLTLSpec,
                 update_intv: int,
                 n_epoch: int,
                 batch_size: int,
                 perf_coef: float,
                 clip_range: float,
                 demo_dataset: np.ndarray = None,
                 opt_kwargs: Dict = None):
        super(Updater, self).__init__()
        self.policy = policy
        self.buffer = buffer
        self.spec = spec
        self.update_intv = update_intv
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.perf_coef = perf_coef
        self.clip_range = clip_range
        self.demo_dataset = demo_dataset
        self.opt_kwargs = opt_kwargs

        if self.opt_kwargs is None:
            self.opt_kwargs = {}
        self.policy_opt = th.optim.Adam(params=self.policy.parameters(),
                                        **self.opt_kwargs)

        self.rollout_count = 0

    def _on_training_start(self) -> None:
        self.logger.set_level(LOG_LEVEL)

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        if self.rollout_count % self.update_intv == 0:
            run_info = self.update()
            self.logger.record("plan_net/pg_loss", run_info["pg_loss"])
            self.logger.record("plan_net/spec_loss", run_info["spec_loss"])
            self.logger.record("plan_net/demo_loss", run_info["demo_loss"])
            self.rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def update(self) -> Dict:
        self.buffer.drop_incomplete()
        buf_indx = np.arange(0, len(self.buffer.trajs))

        if self.demo_dataset is not None:
            demo_indx = np.arange(len(self.demo_dataset))
        else:
            demo_indx = None

        ep_lam = 0

        pg_losses = []
        spec_losses = []
        demo_losses = []

        for _ in range(self.n_epoch):
            np.random.shuffle(buf_indx)
            if demo_indx is not None:
                np.random.shuffle(demo_indx)

            n_batch = len(buf_indx) // self.batch_size
            _lam = 0
            for i in range(n_batch):
                _indx = buf_indx[i * self.batch_size:(i + 1) * self.batch_size]
                batch_trajs = self.buffer.trajs[_indx]
                batch_init_tensor = default_tensor(batch_trajs[:, 0])
                batch_trajs_tensor = self.policy.forward(batch_init_tensor)

                batch_ctrl_rewards = self.buffer.norm_ctrl_rewards[_indx]
                batch_ctrl_rewards_tensor = default_tensor(batch_ctrl_rewards)

                # spec constraint and reward
                spec_loss = default_tensor(0)
                spec_rewards = []
                for traj_tensor in batch_trajs_tensor:
                    # do not support batch now...
                    spec_score = self.spec(traj_tensor)
                    spec_rewards.append(spec_score.detach())
                    spec_loss -= spec_score
                spec_rewards_tensor = th.stack(spec_rewards)

                # clipped PPO policy gradient
                log_prob = self.policy.log_prob(batch_trajs_tensor)
                old_log_prob = self.buffer.log_probs[_indx]
                ratio = th.exp(log_prob - old_log_prob)

                spec_rewards_tensor = spec_rewards_tensor - spec_rewards_tensor.mean()
                spec_rewards_tensor /= spec_rewards_tensor.max() - spec_rewards_tensor.min()

                advantage = spec_rewards_tensor + batch_ctrl_rewards_tensor
                pg_loss_1 = advantage * ratio
                pg_loss_2 = advantage * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = -th.min(pg_loss_1, pg_loss_2).mean()
                pg_losses.append(as_numpy(pg_loss))

                # spec constraint loss
                constraint_loss = spec_loss
                spec_losses.append(as_numpy(spec_loss))

                # demo constraint loss, not required, for fast training only
                if demo_indx is not None:
                    demo_batch_size = max(len(self.demo_dataset) // n_batch, 1)
                    _indx = demo_indx[i * demo_batch_size: (i + 1) * demo_batch_size]
                    batch_demo = self.demo_dataset[_indx]
                    pred_path = self.policy.forward(default_tensor(batch_demo[:, 0, :]))
                    demo_loss = mse_loss(pred_path, default_tensor(batch_demo))
                    constraint_loss += demo_loss
                    demo_losses.append(as_numpy(demo_loss))

                _lam += as_numpy(constraint_loss)
                loss = self.perf_coef * pg_loss + ep_lam * constraint_loss

                self.policy_opt.zero_grad()
                loss.backward()
                self.policy_opt.step()

            ep_lam = _lam
            # ep_lam = max(min(ep_lam, 10), 0.1)

        # on-policy update
        self.buffer.clean()

        return {
            "pg_loss": np.mean(pg_losses),
            "spec_loss": np.mean(spec_losses),
            "demo_loss": np.mean(demo_losses)
        }
