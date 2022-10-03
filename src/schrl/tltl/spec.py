from collections import deque
from typing import List, Union

import numpy as np
import torch as th
from termcolor import colored
from torch.nn.functional import softmax, softmin

from schrl.common.path import Path
from schrl.common.utils import default_tensor
from schrl.tltl.predicate import PredicateBase
from schrl.tltl.subformula import SubFormulaBase


class DiffTLTLSpec:
    def __init__(self, tree: Union[PredicateBase, SubFormulaBase, List], soft: bool = True):
        self.ast = tree
        self.single_operators = ("~", "G", "E", "X")
        self.binary_operators = ("&", "|", "->", "U")
        self.sequence_operators = ("G", "E", "U")
        self.soft = soft

    """
    Syntax Functions
    """

    @classmethod
    def build_from_tree(cls, tree) -> 'DiffTLTLSpec':
        return DiffTLTLSpec(tree)

    def __and__(self, other: 'DiffTLTLSpec') -> 'DiffTLTLSpec':
        ast = ["&", self.ast, other.ast]
        return self.build_from_tree(ast)

    def __or__(self, other: 'DiffTLTLSpec') -> 'DiffTLTLSpec':
        ast = ["|", self.ast, other.ast]
        return self.build_from_tree(ast)

    def __invert__(self) -> 'DiffTLTLSpec':
        ast = ["~", self.ast]
        return self.build_from_tree(ast)

    def implies(self, other: 'DiffTLTLSpec') -> 'DiffTLTLSpec':
        ast = ["->", self.ast, other.ast]
        return self.build_from_tree(ast)

    def next(self):
        ast = ["X", self.ast]
        return self.build_from_tree(ast)

    def eventually(self, start: int = 0, end: int = None):
        ast = ["E", self.ast, start, end]
        return self.build_from_tree(ast)

    def globally(self, start: int = 0, end: int = None) -> 'DiffTLTLSpec':
        ast = ["G", self.ast, start, end]
        return self.build_from_tree(ast)

    def until(self, other: 'DiffTLTLSpec', start: int = 0, end: int = None) -> 'DiffTLTLSpec':
        ast = ["U", self.ast, other.ast, start, end]
        return self.build_from_tree(ast)

    """
    Semantics Functions
    """

    def _eval(self,
              ast: Union[List, PredicateBase, SubFormulaBase],
              traj: Union[Path, th.Tensor]) -> th.Tensor:
        """
        Evaluate an Abstract Syntax Tree (AST) with input trajectory,
        unfold only when requires check all the states in a trajectory (e.g., check global property)
        NOTE: Gradient will break in the right part of until due to the index operator is not differentiable

        :param ast: AST
        :param traj: Trajectory
        :return: score of a trajectory, > 0 means satisfied, < 0 means unsatisfied. Greater value
        means "closer" to satisfied case. This requires user define the predicate as standard.
        """
        if issubclass(type(ast), PredicateBase):
            return self._eval_predicate(ast, traj)
        elif issubclass(type(ast), SubFormulaBase):
            return self._eval_sub_formula(ast, traj)
        elif ast[0] in self.single_operators:
            if ast[0] == "~":
                return self._eval_not(ast, traj)
            elif ast[0] == "G":
                return self._eval_globally(ast, traj)
            elif ast[0] == "E":
                return self._eval_eventually(ast, traj)
            elif ast[0] == "X":
                return self._eval_next(ast, traj)
        elif ast[0] in self.binary_operators:
            if ast[0] == "&":
                return self._eval_and(ast, traj)
            elif ast[0] == "|":
                return self._eval_or(ast, traj)
            elif ast[0] == "->":
                return self._eval_implies(ast, traj)
            elif ast[0] == "U":
                return self._eval_until(ast, traj)

    def _eval_by_unfolding(self,
                           ast: Union[List, PredicateBase, SubFormulaBase],
                           traj: Union[Path, th.Tensor]):
        return [self._eval(ast, traj[i:]) for i in range(len(traj))]

    @staticmethod
    def get_traj_tensor(traj: Union[Path, th.Tensor, np.ndarray]):
        if not isinstance(traj, th.Tensor):
            traj_tensor = traj.to_torch()
        elif isinstance(traj, th.Tensor):
            traj_tensor = traj
        elif isinstance(traj, np.ndarray):
            traj_tensor = default_tensor(traj)
        else:
            raise NotImplementedError(f"type {type(traj)} is not supported")

        return traj_tensor

    def _eval_predicate(self,
                        predicate: PredicateBase,
                        traj: Union[Path, th.Tensor, np.ndarray]) -> th.Tensor:
        """
        One state one value
        :param predicate: PredicateBase
        :param traj: a sequence of state
        :return: a sequence of value with the same length to the input
        """
        traj_tensor = self.get_traj_tensor(traj)
        return predicate(traj_tensor)

    def _eval_sub_formula(self,
                          sub_formula: SubFormulaBase,
                          traj: Union[Path, th.Tensor, np.ndarray]) -> th.Tensor:
        """
        One path one value
        :param sub_formula: SubFormulaBase
        :param traj: a sequence of state
        :return: one value
        """
        traj_tensor = self.get_traj_tensor(traj)
        return sub_formula(traj_tensor)

    def _eval_not(self, ast, traj) -> th.Tensor:
        return -self._eval(ast[1], traj)

    def _eval_and(self, ast, traj) -> th.Tensor:
        left = self._eval(ast[1], traj)
        right = self._eval(ast[2], traj)
        if self.soft:
            return self._soft_min(th.stack([left, right]))
        else:
            return th.minimum(left, right)

    def _eval_or(self, ast, traj) -> th.Tensor:
        left = self._eval(ast[1], traj)
        right = self._eval(ast[2], traj)

        if self.soft:
            return self._soft_max(th.stack([left, right]))
        else:
            return th.maximum(left, right)

    def _eval_implies(self, ast, traj) -> th.Tensor:
        left = self._eval(ast[1], traj)
        right = self._eval(ast[2], traj)

        if self.soft:
            return self._soft_max(th.stack([-left, right]))
        else:
            return th.maximum(-left, right)

    def _eval_next(self, ast, traj):
        # suppose the next of last state is always true
        if len(traj) == 1:
            return th.ones_like(traj[:1, 0])
        res = self._eval(ast[1], traj[1:])
        res = th.cat([res, th.ones_like(res[:1])])
        return res

    def _eval_globally(self, ast, traj) -> th.Tensor:
        if self._is_formula(ast[1]):
            values = self._eval_by_unfolding(ast, traj)
        else:
            values = self._eval(ast[1], traj[ast[-2]:ast[-1]])
        return th.min(values)

    def _eval_eventually(self, ast, traj) -> th.Tensor:
        if self._is_formula(ast[1]):
            values = self._eval_by_unfolding(ast, traj)
        else:
            values = self._eval(ast[1], traj[ast[-2]:ast[-1]])
        return th.max(values)

    def _eval_until(self, ast, traj) -> th.Tensor:
        if self._is_formula(ast[1]):
            left = self._eval_by_unfolding(ast[1], traj)
        else:
            left = self._eval(ast[1], traj[ast[-2]:ast[-1]])

        if self._is_formula(ast[2]):
            right = self._eval_by_unfolding(ast[2], traj)
        else:
            right = self._eval(ast[2], traj[ast[-2]:ast[-1]])

        # This makes the gradient cannot be passed to right part
        until_indx = (right > 0.0).nonzero(as_tuple=False)

        if len(until_indx) > 0:
            until_indx = until_indx[0]
        else:
            until_indx = len(right)

        if until_indx == 0:
            return th.ones_like(left[0])
        else:
            return th.min(left[:until_indx])

    """
    Utilities
    """

    def __call__(self, traj: Union[Path, th.Tensor]) -> th.Tensor:
        res = self._eval(self.ast, traj)

        return res

    @staticmethod
    def _is_leaf(ast):
        return issubclass(type(ast), PredicateBase) or issubclass(type(ast), SubFormulaBase)

    def _is_formula(self, ast):
        if not self._is_leaf(ast):
            if ast[0] in self.sequence_operators:
                return True
            if ast[0] in self.single_operators:
                return self._is_formula(ast[1])
            elif ast[0] in self.binary_operators:
                return self._is_formula(ast[1]) or self._is_formula(ast[2])
        elif issubclass(type(ast), SubFormulaBase):
            return True

        return False

    @staticmethod
    def _soft_max(tensor: th.Tensor, alpha: th.Tensor = default_tensor(2.0), dim: int = 0) -> th.Tensor:
        return th.sum(softmax(tensor * alpha, dim) * tensor, dim=dim)

    @staticmethod
    def _soft_min(tensor: th.Tensor, alpha: th.Tensor = default_tensor(2.0), dim: int = 0) -> th.Tensor:
        return th.sum(softmin(tensor * alpha, dim) * tensor, dim=dim)

    def toggle_soft(self):
        self.soft = True

    def toggle_hard(self):
        self.soft = False

    def __repr__(self):
        single_operators = ("~", "G", "E", "X")
        binary_operators = ("&", "|", "->", "U")
        time_bounded_operators = ("G", "E", "U")

        # traverse ast
        operator_stack: Union[List, PredicateBase, str] = [self.ast]
        expr = ""
        cur = self.ast

        def push_stack(ast):
            if isinstance(ast, str) and ast in time_bounded_operators:
                time_window = f"[{cur[-2]}, {cur[-1]})"
                operator_stack.append(time_window)
            operator_stack.append(ast)

        while operator_stack:
            cur = operator_stack.pop()
            if isinstance(cur, str):
                if cur == "(" or cur == ")":
                    expr += cur
                elif cur.startswith("["):
                    expr += "\b"
                    expr += colored(cur, "yellow")
                else:
                    if cur in ("G", "E", "X"):
                        expr += colored(cur, "magenta") + " "
                    elif cur in ("&", "|", "->", "U"):
                        expr += " " + colored(cur, "magenta") + " "
                    elif cur in ("~",):
                        expr += colored(cur, "magenta")
            elif self._is_leaf(cur):
                expr += str(cur)
            elif cur[0] in single_operators:
                # single operator
                if not self._is_leaf(cur[1]):
                    push_stack(")")
                push_stack(cur[1])
                if not self._is_leaf(cur[1]):
                    push_stack("(")
                push_stack(cur[0])
            elif cur[0] in binary_operators:
                # binary operator
                if not self._is_leaf(cur[2]) and cur[2][0] in binary_operators:
                    push_stack(")")
                    push_stack(cur[2])
                    push_stack("(")
                else:
                    push_stack(cur[2])
                push_stack(cur[0])
                if not self._is_leaf(cur[1]) and cur[1][0] in binary_operators:
                    push_stack(")")
                    push_stack(cur[1])
                    push_stack("(")
                else:
                    push_stack(cur[1])

        return expr

    def get_all_predicates(self):
        all_preds = []
        queue = deque([self.ast])

        while queue:
            cur = queue.popleft()

            if self._is_leaf(cur):
                all_preds.append(cur)
            elif cur[0] in self.single_operators:
                queue.append(cur[1])
            elif cur[0] in self.binary_operators:
                queue.append(cur[1])
                queue.append(cur[2])
            else:
                raise RuntimeError("Should never visit here")

        return all_preds
