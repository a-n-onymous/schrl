from collections import defaultdict
from operator import mod

import numpy as np
import pandas as pd

from schrl.config import DATA_ROOT
from schrl.rebuttal.smoothness.soft_vs_hard import train


def collect_data():
    tasks = [
        "seq",
        "cover",
        "branch",
        "loop",
        "signal"
    ]
    modes = [
        "soft",
        "hard"
    ]
    n_runs = 5
    batch_trajs = 100

    lr_dict = defaultdict(lambda: 1e-2)
    lr_dict["seq"] = 1e-3

    info_dict = {t: defaultdict(list) for t in tasks}

    for t in tasks:
        for m in modes:
            for _ in range(n_runs):
                info_dict[t][m].append(
                    train(t, m, batch_trajs=batch_trajs, lr=lr_dict[t]))
    df = pd.DataFrame(info_dict)
    print(df)
    df.to_csv(f"{DATA_ROOT}/rebuttal/soft_vs_hard.csv")


def count_res():
    df = pd.read_csv(f"{DATA_ROOT}/rebuttal/soft_vs_hard.csv")
    tasks = [
        "seq",
        "cover",
        "branch",
        "loop",
        "signal"
    ]
    modes = [
        "soft",
        "hard"
    ]

    res = defaultdict(list)
    for t in tasks:
        for i, m in enumerate(modes):
            data = [n for n in map(int, df[t][i].strip("[]").split(","))]
            res[t].append(f"{np.mean(data):.2f} ± {np.std(data):.2f}")

    pd.DataFrame.from_dict(res).to_csv(f"{DATA_ROOT}/rebuttal/soft_vs_hard_stat.csv")


if __name__ == '__main__':
    collect_data()
    count_res()
