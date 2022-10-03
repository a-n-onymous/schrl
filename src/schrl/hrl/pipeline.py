from typing import Dict

from schrl.common.utils import wrap_env, as_numpy
from schrl.config import DATA_ROOT, DEFAULT_DEVICE
from schrl.hrl.control_layer import PPOControlLayer
from schrl.hrl.hrl import HRL
from schrl.hrl.plan_layer import LearnablePlanLayer
from schrl.loader import load_general_ctrl_pi
from schrl.tasks import get_task


class Trainer:
    def __init__(self,
                 robot_name: str,
                 task_name: str,
                 enable_gui: bool = False,
                 plan_layer_kwargs: Dict = None,
                 control_layer_kwargs: Dict = None,
                 init_control_pi_type: str = None,
                 load_policy_net_only: bool = False,
                 verbose: int = 1,
                 device: str = DEFAULT_DEVICE):

        self.plan_layer = None
        self.control_layer = None
        self.hrl = None
        if plan_layer_kwargs is None:
            self.plan_layer_kwargs = {}
        else:
            self.plan_layer_kwargs = plan_layer_kwargs

        if control_layer_kwargs is None:
            self.control_layer_kwargs = {}
        else:
            self.control_layer_kwargs = control_layer_kwargs

        self.robot_name = robot_name
        self.task_name = task_name
        self.gui = enable_gui
        self.init_control_pi_type = init_control_pi_type
        self.load_policy_net_only = load_policy_net_only
        self.device = device

        self.task = get_task(robot_name, task_name, enable_gui)
        self.env = self.task.env
        self.venv = wrap_env(self.env, self.task.time_limit, sb3_reward_monitor=True, vec_env=True)
        self.spec = self.task.spec

        self.store_path = f"{DATA_ROOT}/schrl_pi/self_trained/{self.robot_name}/{self.task_name}"

        # stablize training with planned paths
        self.demo_traj_num = self.plan_layer_kwargs.pop("demo_traj_num", 128)
        if self.demo_traj_num > 0:
            self.demo_dataset = self.task.demo(self.demo_traj_num)
        else:
            self.demo_dataset = None

        self.plan_layer = LearnablePlanLayer(self.task,
                                             device=self.device,
                                             demo_dataset=self.demo_dataset,
                                             **self.plan_layer_kwargs)
        self.control_layer = PPOControlLayer("MlpPolicy",
                                             self.venv,
                                             tensorboard_log=f"{self.store_path}/log",
                                             device=self.device,
                                             verbose=verbose,
                                             **self.control_layer_kwargs)

        # warm start control layer from general control policy
        if self.init_control_pi_type is not None:
            self._init_control_policy(self.load_policy_net_only)
        self.hrl = HRL(self.plan_layer, self.control_layer)

    def _init_control_policy(self, policy_only=False):
        if self.init_control_pi_type is not None:
            policy = load_general_ctrl_pi(self.robot_name, self.init_control_pi_type).policy
            if policy_only:
                self.control_layer.algo.policy.mlp_extractor.policy_net.load_state_dict(
                    policy.mlp_extractor.policy_net.state_dict())
                self.control_layer.algo.policy.action_net.load_state_dict(
                    policy.action_net.state_dict()
                )
                self.control_layer.algo.policy.log_std = policy.log_std
            else:
                self.control_layer.algo.policy.load_state_dict(policy.state_dict())

    def learn(self, total_timesteps, log_interval=4):
        self.hrl.learn(total_timesteps, log_interval, tb_log_name="schrl")
        self.hrl.save(f"{self.store_path}/")


class Evaluator:
    def __init__(self,
                 robot_name: str,
                 task_name: str,
                 gui: bool = False):
        self.task = get_task(robot_name, task_name, gui)
        self.hrl = HRL.load(f"{DATA_ROOT}/schrl_pi/self_trained/{robot_name}/{task_name}", env=self.task.env)
        self.env = self.task.env
        self.plan_layer = self.hrl.plan_layer
        self.ctrl_layer = self.hrl.control_layer

    def evaluate(self, max_steps: int, n_ep: int, render: bool = False):
        cum_rewards = []
        cum_steps = []
        for _ in range(n_ep):
            self.env.reset()
            path = self.plan_layer.path_forward(self.env.get_state())[0]
            path = as_numpy(path)
            goal_indx = 0
            self.env.set_goal(path[goal_indx])
            obs = self.env.get_obs()

            cum_reward = 0
            cum_step = 0

            for _ in range(max_steps):
                action, _ = self.ctrl_layer.algo.predict(obs)
                obs, reward, _, info = self.env.step(action)
                cum_reward += reward
                cum_step += 1

                if render:
                    self.env.render()

                if info["reach"]:
                    goal_indx += 1
                    if goal_indx >= len(path):
                        break
                    self.env.set_goal(path[goal_indx])
                    obs = self.env.get_obs()

            cum_rewards.append(cum_reward)
            cum_steps.append(cum_step)

        return cum_steps, cum_rewards
