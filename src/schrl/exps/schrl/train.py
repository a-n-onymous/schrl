from typing import Dict

import yaml

from schrl.config import DATA_ROOT
from schrl.hrl.pipeline import Trainer


def load_hyperparameters(robot_name: str, task_name: str) -> Dict:
    with open(f"{DATA_ROOT}/schrl_pi/hyperparameters.yml", mode="rt", encoding="utf-8") as f:
        return yaml.safe_load(f)[robot_name][task_name]


def train(robot_name: str, task_name: str):
    hp = load_hyperparameters(robot_name, task_name)
    learning_steps = hp.pop("learning_steps")
    trainer = Trainer(robot_name, task_name, **hp)
    trainer.learn(learning_steps)


if __name__ == '__main__':
    train("point", "seq")
