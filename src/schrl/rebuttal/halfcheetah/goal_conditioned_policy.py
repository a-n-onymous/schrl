from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from schrl.config import DATA_ROOT
from schrl.rebuttal.halfcheetah.env import GoalConditionedHalfCheetah


def train_halfcheetah_gc_policy(n_envs: int = 4,
                                total_timesteps: int = 1_000_000,
                                dummy_vec_env=False):
    res_path = f"{DATA_ROOT}/rebuttal/halfcheetah"

    def env_fn():
        _env = GoalConditionedHalfCheetah(continuous_goals=True)
        return Monitor(TimeLimit(_env, 1000))

    vec_env_cls = DummyVecEnv if dummy_vec_env else SubprocVecEnv
    vec_env = vec_env_cls(env_fns=[env_fn] * n_envs)

    algo = PPO("MlpPolicy", vec_env,
               tensorboard_log=f"{res_path}/logs/", verbose=1)
    algo.learn(total_timesteps, tb_log_name=f"gc_pi")

    algo.save(f"{res_path}/models/halfcheetah_gc_policy.zip")
