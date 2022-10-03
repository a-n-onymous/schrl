import argparse

from schrl.envs import make_env
from schrl.loader import load_general_ctrl_pi


def simu(env_name: str,
         algo: str = "ppo",
         max_simu_step: int = 8000,
         n_goals: int = 5,
         enable_gui: bool = True):
    env = make_env(env_name, enable_gui=enable_gui)
    pi = load_general_ctrl_pi(env_name, algo)
    env.reset()

    step_counter = 0
    for _ in range(n_goals):
        env.set_goal(goal=env.goal_space.sample())

        reach = False
        while not reach:
            action, _ = pi.predict(env.get_obs())
            _, _, _, info = env.step(action)
            reach = info.get("reach", False)
            if enable_gui:
                env.render()

            step_counter += 1
            if step_counter > max_simu_step:
                return

    print(f"{env_name} reach {n_goals} goals in {step_counter} steps")


if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='Reach random sampled goals')
    paser.add_argument("--robot_name", type=str,
                       default="doggo", help="robot name")
    paser.add_argument("--algo", type=str, default="ppo",
                       help="controller trained by this RL algorithm")
    paser.add_argument("--max_simu_steps", type=int,
                       default=8000, help="max simulation steps")
    paser.add_argument("--n_goals", type=int, default=5,
                       help="number of goals to reach")
    paser.add_argument("--gui", action="store_true", help="enable gui")

    args = paser.parse_args()
    simu(args.robot_name,
         algo=args.algo,
         max_simu_step=args.max_simu_steps,
         n_goals=args.n_goals,
         enable_gui=args.gui)
