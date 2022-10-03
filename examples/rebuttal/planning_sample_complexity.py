import numpy as np
import pandas as pd

from schrl.config import DATA_ROOT
from schrl.rebuttal.planning_sample_complexity.learning import SpecConstrainedPG
from schrl.rebuttal.tasks import get_task


def with_constrained_learning(task_name="seq"):
    task = get_task(task_name)
    algo = SpecConstrainedPG(task, n_trajs=100)
    term_n_samples = algo.learn(10_000)
    res_path = f"{DATA_ROOT}/rebuttal/planning_sample_complexity/{task_name}-constrained.csv"
    with open(res_path, "a") as f:
        f.write(f"{term_n_samples},")


def without_constrained_learning(task_name="seq"):
    task = get_task(task_name)
    algo = SpecConstrainedPG(task, n_trajs=100, constrained=False)
    term_n_samples = algo.learn(10_000)
    res_path = f"{DATA_ROOT}/rebuttal/planning_sample_complexity/{task_name}-pg.csv"
    with open(res_path, "a") as f:
        f.write(f"{term_n_samples},")


def collect_data():
    tasks = [
        "seq",
        "cover",
        "branch",
        "loop",
        "signal"
    ]
    for _ in range(5):
        for task in tasks:
            with_constrained_learning(task)
            without_constrained_learning(task)


def read_file(path: str):
    with open(path) as f:
        data = f.readline()
    return np.array([float(n) for n in data[:-1].split(",")])


def count_res():
    tasks = [
        "seq",
        "cover",
        "branch",
        "loop",
        "signal"
    ]

    for t in tasks:
        pg_res = read_file(
            f"{DATA_ROOT}/rebuttal/planning_sample_complexity/{t}-pg.csv")
        constrainted_res = read_file(
            f"{DATA_ROOT}/rebuttal/planning_sample_complexity/{t}-constrained.csv")
        print(
            f"{pg_res[pg_res<10000].mean():.2f} ± {pg_res[pg_res<10000].std():.2f} ({sum(pg_res<10000)} / {len(pg_res)})"
            f"\t|\t{constrainted_res.mean():.2f} ± {constrainted_res.std():.2f} ({sum(constrainted_res<10000)} / {len(constrainted_res)})")


if __name__ == '__main__':
    collect_data()
    count_res()
