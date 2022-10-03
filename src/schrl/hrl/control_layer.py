from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from schrl.hrl.sb3_algo import SB3PPO


class ControlLayerBase(ABC):
    def __init__(self, algo: BaseAlgorithm):
        self.algo = algo
        self.callback_list = None

    def register_callbacks(self, callback_list: CallbackList):
        self.callback_list = callback_list

    def learn(self,
              total_timesteps: int,
              log_interval: int = 4,
              *args, **kwargs):
        self.algo.learn(total_timesteps=total_timesteps,
                        log_interval=log_interval,
                        callback=self.callback_list,
                        *args, **kwargs)

    @abstractmethod
    def disable_learning(self):
        pass

    @abstractmethod
    def enable_learning(self):
        pass

    def save(self, path: str):
        self.algo.save(f"{path}/controller")

    @classmethod
    @abstractmethod
    def load(cls, path: str, env: GymEnv):
        pass


class PPOControlLayer(ControlLayerBase):
    def __init__(self,
                 policy: Union[str, Type[ActorCriticPolicy]] = None,
                 env: Union[GymEnv, str] = None,
                 learning_rate: Union[float, Schedule] = 3e-4,
                 n_steps: int = 4000,
                 batch_size: int = 100,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 clip_range_vf: Union[None, float, Schedule] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,
                 _init_algo=True):
        algo = None
        if _init_algo:
            algo = SB3PPO(
                policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda,
                clip_range, clip_range_vf, normalize_advantage, ent_coef, vf_coef, max_grad_norm,
                use_sde, sde_sample_freq, target_kl, tensorboard_log, create_eval_env,
                policy_kwargs, verbose, seed, device, _init_setup_model)
        super(PPOControlLayer, self).__init__(algo)

    def set_policy(self, policy):
        self.algo.policy.load_state_dict(policy.state_dict())

    def disable_learning(self):
        self.algo.disable_training()

    def enable_learning(self):
        self.algo.enable_training()

    @classmethod
    def load(cls, path: str, env: GymEnv, device: str = "auto"):
        algo = SB3PPO.load(f"{path}/controller", env=env, device=device)
        ppo_control_layer = cls(_init_algo=False)
        ppo_control_layer.algo = algo

        return ppo_control_layer
