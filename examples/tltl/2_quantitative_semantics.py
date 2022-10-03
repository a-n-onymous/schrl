import numpy as np
import torch as th
from termcolor import colored

from schrl.common.path import Path
from schrl.tltl.predicate import ProgrammablePredicate
from schrl.tltl.spec import DiffTLTLSpec


def _assert(spec: DiffTLTLSpec, traj: Path, result: bool):
    confidence = spec(traj)

    if len(confidence.shape) > 0 and len(confidence) > 1:
        # only check current state
        confidence = confidence[0]

    msg = f"{colored(str(traj), 'yellow')} {colored('|=', 'red')} " \
          f"{spec} = {colored(str(result), 'green')}"

    if (confidence > 0) == result:
        print(f"[OK] {msg}")
    else:
        error = f"\nExpect \n" \
                f"{msg} \n" \
                f"but got \n" \
                f"{colored(str(traj), 'yellow')} {colored('|=', 'red')} " \
                f"{spec} = {colored(str(confidence.detach().cpu().numpy().item() > 0), 'green')} " \
                f"(confidence: {confidence})"
        assert False, error


def _optimize(spec: DiffTLTLSpec,
              traj: Path,
              param,
              step: int = 100,
              expected_result: bool = True):
    initial_obj = spec(traj)
    initial_param = param.detach().clone()

    for i in range(step):
        obj = spec(traj)
        if len(obj.shape) > 0 and len(obj) > 1:
            # only check current state
            obj = obj[0]
        obj.backward()
        if expected_result:
            param.data += 1e-2 * param.grad
        else:
            param.data -= 1e-2 * param.grad

    to_num = lambda x: str(x.detach().cpu().numpy())
    print(f"\t predicate value: {colored(to_num(initial_obj), 'blue')} -> "
          f"{colored(to_num(spec(traj)), 'cyan')}")
    print(f"\t parameter: \t \t {colored(str(to_num(initial_param)), 'blue')} -> "
          f"{colored(to_num(param), 'cyan')}")


def test_forward_not():
    geq_10 = ProgrammablePredicate(lambda x: x - 10, "geq_10")
    spec = ~DiffTLTLSpec(geq_10)
    traj1 = Path(np.array([9, 9, 9]))
    traj2 = Path(np.array([11, 9.9, 9.9]))

    _assert(spec, traj1, True)
    _assert(spec, traj2, False)


def test_forward_and():
    geq_10 = ProgrammablePredicate(lambda x: x - 10, "geq_10")
    leq_11 = ProgrammablePredicate(lambda x: 11 - x, "leq_11")
    spec = DiffTLTLSpec(geq_10) & DiffTLTLSpec(leq_11)
    traj1 = Path(np.array([9, 9, 9]))
    traj2 = Path(np.array([10.5]))

    _assert(spec, traj1, False)
    _assert(spec, traj2, True)


def test_forward_or():
    leq_10 = ProgrammablePredicate(lambda x: 10 - x, "geq_10")
    geq_11 = ProgrammablePredicate(lambda x: x - 11, "geq_11")
    spec = DiffTLTLSpec(leq_10) | DiffTLTLSpec(geq_11)
    traj1 = Path(np.array([9, 9, 9]))
    traj2 = Path(np.array([10.5]))
    traj3 = Path(np.array([11.5, 9]))

    _assert(spec, traj1, True)
    _assert(spec, traj2, False)
    _assert(spec, traj3, True)


def test_forward_imply():
    leq_10 = ProgrammablePredicate(lambda x: 10 - x, "leq_10")
    geq_11 = ProgrammablePredicate(lambda x: x - 11, "geq_11")
    spec = DiffTLTLSpec(leq_10).implies(DiffTLTLSpec(geq_11).next())
    traj1 = Path(np.array([9, 12]))
    traj2 = Path(np.array([9, 10]))

    _assert(spec, traj1, True)
    _assert(spec, traj2, False)


def test_forward_next():
    leq_10 = ProgrammablePredicate(lambda x: 10 - x, "geq_10")
    spec = DiffTLTLSpec(leq_10).next()
    traj1 = Path(np.array([9, 9, 9]))
    traj2 = Path(np.array([8, 11, 9]))
    traj3 = Path(np.array([10.5, 9, 5]))
    traj4 = Path(np.array([9.5, 11]))

    _assert(spec, traj1, True)
    _assert(spec, traj2, False)
    _assert(spec, traj3, True)
    _assert(spec, traj4, False)


def test_forward_globally():
    leq_10 = ProgrammablePredicate(lambda x: 10 - x, "leq_10")
    spec = (DiffTLTLSpec(leq_10)).globally()
    traj1 = Path(np.array([11] * 10))
    traj2 = Path(np.array([9] * 10))

    _assert(spec, traj1, False)
    _assert(spec, traj2, True)


def test_forward_eventually():
    leq_10 = ProgrammablePredicate(lambda x: 10 - x, "leq_10")
    spec = (DiffTLTLSpec(leq_10)).eventually()

    traj1 = Path(np.array([11] * 3))
    traj2 = Path(np.array([11] * 3 + [9]))
    traj3 = Path(np.array([11] * 3 + [9, 10]))

    _assert(spec, traj1, False)
    _assert(spec, traj2, True)
    _assert(spec, traj3, True)


def test_forward_until():
    leq_10 = ProgrammablePredicate(lambda x: 10 - x, "leq_10")
    geq_11 = ProgrammablePredicate(lambda x: x - 11, "geq_11")
    spec = DiffTLTLSpec(leq_10).until(DiffTLTLSpec(geq_11))
    traj1 = Path(np.array([9] * 3 + [12, 12, 13]))
    traj2 = Path(np.array([9] * 3 + [10.5, 12, 12]))
    traj3 = Path(np.array([20, 20]))

    _assert(spec, traj1, True)
    _assert(spec, traj2, False)
    _assert(spec, traj3, True)


def test_backward_not():
    param = th.tensor(7.0, requires_grad=True)
    x_geq_param = ProgrammablePredicate(lambda x: x - param, "(x > param)")
    spec = ~DiffTLTLSpec(x_geq_param)
    traj = Path(np.array([8, 9, 2]))

    _assert(spec, traj, False)
    _optimize(spec, traj, param)
    _assert(spec, traj, True)


def test_backward_and():
    param = th.tensor([7.0, 10.0], requires_grad=True)
    x_geq_param1 = ProgrammablePredicate(lambda x: x - param[0], "(x > param1)")
    x_leq_param2 = ProgrammablePredicate(lambda x: param[1] - x, "(x < param2)")
    spec = DiffTLTLSpec(x_geq_param1) & DiffTLTLSpec(x_leq_param2)
    traj = Path(np.array([5, 9, 2]))

    _assert(spec, traj, False)
    _optimize(spec, traj, param)
    _assert(spec, traj, True)


def test_backward_or():
    param = th.tensor([10.0, 7.0], requires_grad=True)
    x_geq_param1 = ProgrammablePredicate(lambda x: x - param[0], "(x > param1)")
    x_leq_param2 = ProgrammablePredicate(lambda x: param[1] - x, "(x < param2)")
    spec = DiffTLTLSpec(x_geq_param1) | DiffTLTLSpec(x_leq_param2)
    traj = Path(np.array([8, 9, 2]))

    _assert(spec, traj, False)
    _optimize(spec, traj, param)
    _assert(spec, traj, True)


def test_backward_imply():
    param = th.tensor([6.0, 5.0], requires_grad=True)
    x_geq_param1 = ProgrammablePredicate(lambda x: x - param[0], "(x > param1)")
    x_leq_param2 = ProgrammablePredicate(lambda x: param[1] - x, "(x < param2)")
    spec = DiffTLTLSpec(x_geq_param1).implies(DiffTLTLSpec(x_leq_param2).next())
    traj = Path(np.array([8, 9, 2]))

    _assert(spec, traj, False)
    _optimize(spec, traj, param)
    _assert(spec, traj, True)


def test_backward_next():
    param = th.tensor(10.0, requires_grad=True)
    x_geq_param = ProgrammablePredicate(lambda x: x - param, "(x > param)")
    spec = DiffTLTLSpec(x_geq_param).next()
    traj = Path(np.array([10, 9, 9]))

    _assert(spec, traj, False)
    _optimize(spec, traj, param)
    _assert(spec, traj, True)


def test_backward_globally():
    param = th.tensor(10.0, requires_grad=True)
    x_geq_param = ProgrammablePredicate(lambda x: x - param, "(x > param)")
    spec = DiffTLTLSpec(x_geq_param).globally()
    traj = Path(np.array([9, 8, 9]))

    _assert(spec, traj, False)
    _optimize(spec, traj, param)
    _assert(spec, traj, True)


def test_backward_eventually():
    param = th.tensor(10.0, requires_grad=True)
    x_geq_param = ProgrammablePredicate(lambda x: x - param, "(x > param)")
    spec = DiffTLTLSpec(x_geq_param).eventually()
    traj = Path(np.array([9, 8, 9]))

    _assert(spec, traj, False)
    _optimize(spec, traj, param)
    _assert(spec, traj, True)


def test_backward_until():
    param = th.tensor([10.0, 8.5], requires_grad=True)
    x_geq_param1 = ProgrammablePredicate(lambda x: x - param[0], "(x > param1)")
    x_leq_param2 = ProgrammablePredicate(lambda x: x - param[1], "(x > param2)")
    spec = DiffTLTLSpec(x_geq_param1).until(DiffTLTLSpec(x_leq_param2))
    traj = Path(np.array([8, 9, 2]))

    _assert(spec, traj, False)
    _optimize(spec, traj, param)
    _assert(spec, traj, True)


if __name__ == '__main__':
    test_forward_not()
    test_forward_and()
    test_forward_or()
    test_forward_imply()
    test_forward_next()
    test_forward_globally()
    test_forward_eventually()
    test_forward_until()

    test_backward_not()
    test_backward_and()
    test_backward_or()
    test_backward_imply()
    test_backward_next()
    test_backward_globally()
    test_backward_eventually()
    test_backward_until()
