import argparse

from schrl.exps.schrl.evaluate import evaluate_hrl_policy
from schrl.loader import load_schrl_pi


def inspect_pretrained_policy(robot_name: str,
                              task_name: str,
                              enable_gui: bool = True,
                              n_eps: int = 1):
    schrl_pi = load_schrl_pi(robot_name, task_name, enable_gui)
    eval_data = evaluate_hrl_policy(schrl_pi, n_eps)
    print(eval_data)


if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='Pretrained Policy')
    paser.add_argument("--robot_name", type=str,
                       default="point", help="robot name")
    paser.add_argument("--task_name", type=str,
                       default="seq", help="task name")
    paser.add_argument("--gui", action="store_true", help="enable gui")
    paser.add_argument("--n_eps", type=int, default=5,
                       help="the number of epochs")

    args = paser.parse_args()
    inspect_pretrained_policy(
        args.robot_name, args.task_name, args.gui, args.n_eps)
