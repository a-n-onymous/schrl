import numpy as np
import pandas as pd

from schrl.config import DATA_ROOT
from schrl.rebuttal.planning_time.gradient import grad_plan
from schrl.rebuttal.planning_time.mip import mip_plan
from schrl.rebuttal.planning_time.planning_net import net_plan


def eval_plan_time():
    tasks = [
        "seq",
        "cover",
        "branch",
        "loop",
        "signal"
    ]
    all_dfs = {}
    for t in tasks:
        net_plan_times = net_plan(t)
        mip_times = [mip_plan(t) for _ in range(100)]
        grad_plan_times = grad_plan(t)
        print(f"{t} done")

        task_df = pd.DataFrame(
            np.array([net_plan_times, mip_times, grad_plan_times]).T, columns=["net", "mip", "grad"])
        all_dfs[t] = task_df

    df_list = []
    for k in all_dfs:
        all_dfs[k]["task"] = k
        df_list.append(all_dfs[k])
    df = pd.concat(df_list)
    print(df.groupby("task").mean())
    df.to_csv(f"{DATA_ROOT}/rebuttal/planning_time/planning_times.csv")


if __name__ == '__main__':
    eval_plan_time()
