import torch as th
from stable_baselines3 import PPO

from schrl.config import DATA_ROOT
from schrl.hrl.policy import GruOdePolicy
from schrl.rebuttal.halfcheetah.env import GoalConditionedHalfCheetah


def simu_with_schrl_pi(spec: str):
    if spec == "M1":
        plan_policy = GruOdePolicy(1, 5)
    elif spec == "M2":
        plan_policy = GruOdePolicy(1, 9)
    else:
        raise NotImplementedError()

    plan_policy.load_state_dict(th.load(f"{DATA_ROOT}/rebuttal/halfcheetah/models/{spec}/plan_net.pth"))
    ctrl_policy = PPO.load(f"{DATA_ROOT}/rebuttal/halfcheetah/models/{spec}/controller.zip")
    env = GoalConditionedHalfCheetah(continuous_goals=False)

    max_simu_step = 10000
    step_counter = 0

    path = plan_policy.predict(env.get_state())

    for wp in path:
        env.set_goal(goal=wp)
        print(f"new goal: {env.get_goal()}")

        reach = False
        while not reach:
            action, _ = ctrl_policy.predict(env.get_obs())
            _, _, _, info = env.step(action)
            reach = info.get("reach", False)
            env.render()

            step_counter += 1
            if step_counter > max_simu_step:
                return

    print(f"halfcheetah reach all planned goals in {step_counter} steps")


if __name__ == "__main__":
    simu_with_schrl_pi("M1")
    simu_with_schrl_pi("M2")
