from functools import reduce
from typing import List

import torch as th

from schrl.common.nn import MLP
from schrl.common.types import Vec
from schrl.common.utils import default_tensor
from schrl.config import DATA_ROOT, DEFAULT_DEVICE
from schrl.tltl.predicate import NeuralPredicate, ProgrammablePredicate
from schrl.tltl.spec import DiffTLTLSpec
from schrl.tltl.subformula import ProgrammableSubFormula


class CloseFn:
    def __init__(self, center: Vec, close_radius: float):
        self.center_tensor = default_tensor(center)
        self.close_radius = close_radius

    def __call__(self, x):
        if len(x) == 0:
            # empty state means trajectory does not have length required in formula
            # this can only be caused by incomplete trajectory
            # set this case as false
            return default_tensor(-1.0)
        return self.close_radius - th.norm(x - self.center_tensor, dim=-1)


def ith_state(state: Vec, i, close_radius: float = 0.3) -> DiffTLTLSpec:
    """
    i-th state in a trajectory should be close to the given state
    :param state: i-th state should get close to this state
    :param i: state index in trajectory
    :param close_radius: how close is enough
    :return: DiffTLTLSpec
    """
    close_fn = CloseFn(center=state, close_radius=close_radius)
    formula = ProgrammableSubFormula(lambda traj: close_fn(traj[i]),
                                     f"traj[{i}]<<{state}>>")
    spec = DiffTLTLSpec(formula)

    return spec


def start_state(start: Vec, close_radius: float = 0.3) -> DiffTLTLSpec:
    return ith_state(start, 1, close_radius)


def end_state(end: Vec, close_radius=0.3) -> DiffTLTLSpec:
    return ith_state(end, -1, close_radius)


def dim_lb(lb: float = 0.0, dim=-1) -> DiffTLTLSpec:
    """
    One state dimension should always be greater than lb for all the states in the given trajectory
    :param lb: lower bound
    :param dim: dimension
    :return: DiffTLTLSpec
    """

    def lb_fn(x):
        return th.min(x[:, dim]) - lb

    pred = ProgrammablePredicate(lb_fn, f"lower_bound<<{lb} on dim-{dim}>>")
    spec = DiffTLTLSpec(pred)

    return spec


def dim_ub(ub: float = 0.0, dim=-1) -> DiffTLTLSpec:
    """
    One state dimension should always be smaller than ub for all the states in the given trajectory
    :param ub: lower bound
    :param dim: dimension
    :return: DiffTLTLSpec
    """

    def ub_fn(x):
        return ub - th.max(x[:, dim])

    pred = ProgrammablePredicate(ub_fn, f"upper_bound<<{ub} on dim-{dim}>>")
    spec = DiffTLTLSpec(pred)

    return spec


def step_limit(radius: float) -> DiffTLTLSpec:
    """
    Limit the step size between two sequential waypoints
    :param radius: radius
    :return: DiffTLTLSpec
    """

    def ctrl_limit_fn(traj: th.Tensor) -> th.Tensor:
        assert len(traj.shape) == 2, "input must be a trajectory"
        return (radius - th.norm(traj[1:] - traj[:-1], dim=-1)).min()

    ctrl_limit_pred = ProgrammableSubFormula(ctrl_limit_fn, f"<<r={radius}>>ctrl_limit")
    spec = DiffTLTLSpec(ctrl_limit_pred)
    return spec


def neural_obstacle(obs_name: str,
                    predicate_name: str = "neural_obstacle",
                    device: str = DEFAULT_DEVICE) -> DiffTLTLSpec:
    """
    neural network obstacle spec
    :param obs_name: only support wall and duck for now
    :param predicate_name: name when printing the spec
    :param device: device
    :return: DiffTLTLSpec
    """
    mlp = MLP(3, 1)
    nn_path = f"{DATA_ROOT}/nn_predicates/obstacles/{obs_name}.pth"
    mlp.load_state_dict(th.load(nn_path))
    mlp.to(device)

    pred = NeuralPredicate(mlp, predicate_name)
    spec = DiffTLTLSpec(pred).globally()

    return spec


def sequence(
        goals: Vec,
        seq_time: List[List],
        close_radius: float = 0.3,
        name_prefix: str = "goal") -> DiffTLTLSpec:
    """
    Sequential goal specification
    :param goals: goal in sequence
    :param seq_time: time slots for reach each goal
    :param close_radius: how close is close
    :param name_prefix: name prefix when printing the spec
    :return: DiffTLTLSpec
    """
    predicates = [ProgrammablePredicate(CloseFn(wp, close_radius), f"{name_prefix}-{i}") for i, wp in
                  enumerate(goals)]
    wp_specs = [DiffTLTLSpec(p).eventually(*t) for p, t in zip(predicates, seq_time)]
    seq_sepc = reduce(lambda a, b: a & b, wp_specs)

    return seq_sepc


def coverage(goals: Vec,
             close_radius: float = 0.3,
             name_prefix: str = "goal") -> DiffTLTLSpec:
    """
    reach all the goals and ignore the order
    :param goals: goals to reach
    :param close_radius: how close is close
    :param name_prefix: name prefix when printing the spec
    :return: DiffTLTLSpec
    """
    predicates = [ProgrammablePredicate(CloseFn(wp, close_radius), f"{name_prefix}-{i}") for i, wp in
                  enumerate(goals)]
    wp_specs = [DiffTLTLSpec(p).eventually() for p in predicates]
    cover_spec = reduce(lambda a, b: a & b, wp_specs)

    return cover_spec


def branch(branches: List, close_radius: float = 0.3, name_prefix: str = "goal") -> DiffTLTLSpec:
    all_branches = []

    for i, b in enumerate(branches):
        cond = ith_state(b[0], 1, close_radius)
        res = DiffTLTLSpec(ProgrammablePredicate(
            CloseFn(b[1], close_radius), f"{name_prefix}-{i}")).eventually(start=1)
        all_branches.append(cond.implies(res) & cond & res)

    branch_spec = reduce(lambda b1, b2: b1 | b2, all_branches)

    return branch_spec


def loop(waypoints: Vec,
         close_radius: float = 0.3,
         name_prefix: str = "loop-wp") -> DiffTLTLSpec:
    """
    loop spec
    :param waypoints: loop between these waypoints
    :param close_radius: how close is close
    :param name_prefix: name prefix when printing the spec
    :return: DiffTLTLSpec
    """

    specs = [
        DiffTLTLSpec(
            ProgrammablePredicate(
                CloseFn(goal, close_radius), f"{name_prefix}-{i}"))
        for i, goal in enumerate(waypoints)
    ]
    loop_specs = [specs[i] & specs[(i + 1) % len(specs)].next() for i in range(len(specs))]
    loop_spec = reduce(lambda sp1, sp2: sp1 | sp2, loop_specs).globally(start=1)
    return loop_spec


def loop_alt(waypoints: Vec,
             close_radius: float = 0.3,
             name_prefix: str = "loop-wp") -> DiffTLTLSpec:
    """
    loop spec
    :param waypoints: loop between these waypoints
    :param close_radius: how close is close
    :param name_prefix: name prefix when printing the spec
    :return: DiffTLTLSpec
    """
    specs = []

    for i, goal in enumerate(waypoints):
        pred = ProgrammablePredicate(CloseFn(goal, close_radius), f"{name_prefix}-{i}")
        specs.append(DiffTLTLSpec(pred))

    loop_spec = None
    for i in range(len(specs)):
        if loop_spec is None:
            loop_spec = specs[i].implies(~specs[i].next())
        else:
            loop_spec = loop_spec & specs[i].implies(~specs[i].next())

    reach_spec = None
    for sp in specs:
        if reach_spec is None:
            reach_spec = sp
        else:
            reach_spec = reach_spec | sp

    loop_spec = (loop_spec & reach_spec).globally(start=1)
    return loop_spec


def signal(waypoints: Vec,
           final_goal: Vec,
           until_time: int,
           close_radius: float = 0.3,
           name_prefix: str = "loop-wp") -> DiffTLTLSpec:
    """
    Signal spec
    :param waypoints: loop between these waypoints until signal
    :param final_goal: final goal after signal
    :param until_time: loop until this time
    :param close_radius: how close is close
    :param name_prefix: name prefix
    :return: DiffTLTLSpec
    """
    specs = []

    def until_fn(traj):
        assert len(traj) == until_time + 1, f"len(traj): {len(traj)}, until_time: {until_time}"
        until_cond_tensor = th.ones(len(traj))
        # minus one because until starts from 1s
        until_cond_tensor[:until_time - 1] = 0

        return until_cond_tensor

    specs = [DiffTLTLSpec(ProgrammablePredicate(CloseFn(goal, close_radius), f"{name_prefix}-{i}"))
             for i, goal in enumerate(waypoints)]
    loop_specs = [specs[i] & specs[(i + 1) % len(specs)].next() for i in range(len(specs))]
    loop_spec = reduce(lambda sp1, sp2: sp1 | sp2, loop_specs)

    until_cond = DiffTLTLSpec(ProgrammablePredicate(until_fn, f"{until_time}s"))
    spec = end_state(final_goal, close_radius=close_radius) & loop_spec.until(until_cond, start=1)

    return spec
