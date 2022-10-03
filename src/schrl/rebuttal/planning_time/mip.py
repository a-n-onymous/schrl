# STL planning script from https://github.com/sundw2014/STLPlanning
import time

import numpy as np
from gurobipy import *

from schrl.rebuttal.tasks import Branch, Cover, Seq, Loop, Signal
from schrl.tasks.point import *

M = 1e3
# a large M causes numerical issues and make the model infeasible to Gurobi
T_MIN_SEP = 1e-2
# see comments in GreaterThanZero
IntFeasTol = 1e-1 * T_MIN_SEP / M


def setM(v):
    global M, IntFeasTol
    M = v
    IntFeasTol = 1e-1 * T_MIN_SEP / M


EPS = 1e-2


def _sub(x1, x2):
    return [x1[i] - x2[i] for i in range(len(x1))]


def _add(x1, x2):
    return [x1[i] + x2[i] for i in range(len(x1))]


def L1Norm(model, x):
    xvar = model.addVars(len(x), lb=-GRB.INFINITY)
    abs_x = model.addVars(len(x))
    model.update()
    xvar = [xvar[i] for i in range(len(xvar))]
    abs_x = [abs_x[i] for i in range(len(abs_x))]
    for i in range(len(x)):
        model.addConstr(xvar[i] == x[i])
        model.addConstr(abs_x[i] == abs_(xvar[i]))
    return sum(abs_x)


class Conjunction(object):
    # conjunction node
    def __init__(self, deps=[]):
        super(Conjunction, self).__init__()
        self.deps = deps
        self.constraints = []


class Disjunction(object):
    # disjunction node
    def __init__(self, deps=[]):
        super(Disjunction, self).__init__()
        self.deps = deps
        self.constraints = []


def noIntersection(a, b, c, d):
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b < c or d < a
    return Disjunction([c - b - EPS, a - d - EPS])


def hasIntersection(a, b, c, d):
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b >= c and d >= a
    return Conjunction([b - c, d - a])


def always(i, a, b, zphis, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    conjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        conjunctions.append(Disjunction(
            [noIntersection(t_j, t_j_1, t_i + a, t_i_1 + b), zphis[j]]))
    return Conjunction(conjunctions)


def next(i, zphis, PWL, extra=1):
    # Next PWL segment should satisfy zphi
    if i + extra < len(PWL) - 1:
        return zphis[i + extra]
    else:
        return zphis[-1]


def eventually(i, a, b, zphis, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    z_intervalWidth = b - a - (t_i_1 - t_i) - EPS
    disjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        disjunctions.append(Conjunction(
            [hasIntersection(t_j, t_j_1, t_i + a, t_i_1 + b), zphis[j]]))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])


def bounded_eventually(i, a, b, zphis, PWL, tmax):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    z_intervalWidth = b - a - (t_i_1 - t_i) - EPS
    disjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        disjunctions.append(Conjunction(
            [hasIntersection(t_j, t_j_1, t_i + a, t_i_1 + b), zphis[j]]))
    return Disjunction(
        [Conjunction([z_intervalWidth, Disjunction(disjunctions)]), t_i + b - tmax - EPS])


def until(i, a, b, zphi1s, zphi2s, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    z_intervalWidth = b - a - (t_i_1 - t_i) - EPS
    disjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        conjunctions = [hasIntersection(
            t_j, t_j_1, t_i_1 + a, t_i + b), zphi2s[j]]
        for l in range(j + 1):
            t_l = PWL[l][1]
            t_l_1 = PWL[l + 1][1]
            conjunctions.append(Disjunction(
                [noIntersection(t_l, t_l_1, t_i, t_i_1 + b), zphi1s[l]]))
        disjunctions.append(Conjunction(conjunctions))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])


def release(i, a, b, zphi1s, zphi2s, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    conjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        disjunctions = [noIntersection(
            t_j, t_j_1, t_i_1 + a, t_i + b), zphi2s[j]]
        for l in range(j):
            t_l = PWL[l][1]
            t_l_1 = PWL[l + 1][1]
            disjunctions.append(Conjunction(
                [hasIntersection(t_l, t_l_1, t_i_1, t_i_1 + b), zphi1s[l]]))
        conjunctions.append(Disjunction(disjunctions))
    return Conjunction(conjunctions)


def mu(i, PWL, bloat_factor, A, b):
    bloat_factor = np.max([0, bloat_factor])
    # this segment is fully contained in Ax<=b (shrinked)
    b = b.reshape(-1)
    num_edges = len(b)
    conjunctions = []
    for e in range(num_edges):
        a = A[e, :]
        for j in [i, i + 1]:
            x = PWL[j][0]
            conjunctions.append(b[e] - np.linalg.norm(a) * bloat_factor -
                                sum([a[k] * x[k] for k in range(len(x))]) - EPS)
    return Conjunction(conjunctions)


def negmu(i, PWL, bloat_factor, A, b):
    # this segment is outside Ax<=b (bloated)
    b = b.reshape(-1)
    num_edges = len(b)
    disjunctions = []
    for e in range(num_edges):
        a = A[e, :]
        conjunctions = []
        for j in [i, i + 1]:
            x = PWL[j][0]
            conjunctions.append(sum([a[k] * x[k] for k in range(len(x))]) -
                                (b[e] + np.linalg.norm(a) * bloat_factor) - EPS)
        disjunctions.append(Conjunction(conjunctions))
    return Disjunction(disjunctions)


def add_space_constraints(model, xlist, limits, bloat=0.):
    xlim, ylim = limits
    for x in xlist:
        model.addConstr(x[0] >= (xlim[0] + bloat))
        model.addConstr(x[1] >= (ylim[0] + bloat))
        model.addConstr(x[0] <= (xlim[1] - bloat))
        model.addConstr(x[1] <= (ylim[1] - bloat))

    return None


def add_time_constraints(model, PWL, tmax=None):
    if tmax is not None:
        model.addConstr(PWL[-1][1] <= tmax - T_MIN_SEP)

    for i in range(len(PWL) - 1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i + 1]
        model.addConstr(t2 - t1 >= T_MIN_SEP)


def add_velocity_constraints(model, PWL, vmax=3):
    for i in range(len(PWL) - 1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i + 1]
        # squared_dist = sum([(x1[j]-x2[j])*(x1[j]-x2[j]) for j in range(len(x1))])
        # model.addConstr(squared_dist <= (vmax**2) * (t2 - t1) * (t2 - t1))
        L1_dist = L1Norm(model, _sub(x1, x2))
        model.addConstr(L1_dist <= vmax * (t2 - t1))


def disjoint_segments(model, seg1, seg2, bloat):
    assert (len(seg1) == 2)
    assert (len(seg2) == 2)
    # assuming that bloat is the error bound in two norm for one agent
    return 0.5 * L1Norm(model, _sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1]))) - 0.5 * (
            L1Norm(model, _sub(seg1[0], seg1[1])) + L1Norm(model, _sub(seg2[0], seg2[
        1]))) - 2 * bloat * np.sqrt(
        len(seg1[0])) - EPS


def add_simple_time_constr(model, t11, t12, t21, t22):
    # a simmple version of the time disjoint constraint assuming t1 after t2
    print("forcing {} > {}".format(t11, t22))
    model.addConstr(t11 - t22 >= T_MIN_SEP)
    # model.addConst(t12 - t22 - EPS > 0)


def add_mutual_clearance_constraints(model, PWLs, bloat):
    for i in range(len(PWLs)):
        for j in range(i + 1, len(PWLs)):
            PWL1 = PWLs[i]
            PWL2 = PWLs[j]
            for k in range(1, len(PWL1) - 1):
                for l in range(1, len(PWL2) - 1):
                    x11, t11 = PWL1[k]
                    x12, t12 = PWL1[k + 1]
                    x21, t21 = PWL2[l]
                    x22, t22 = PWL2[l + 1]
                    # TODO: CHECK NO INTERSECTION IN TIME SOLN NOT FOUND???
                    # print("agent {} to {} pwl {} and {}".format(i,j,k,l),t11,t12,t21,t22)
                    z_noIntersection = noIntersection(t11, t12, t21, t22)
                    # z = z_noIntersection
                    z_disjoint_segments = disjoint_segments(
                        model, [x11, x12], [x21, x22], bloat)
                    z = Disjunction([z_noIntersection, z_disjoint_segments])
                    add_CDTree_Constraints(model, z)
                    # add_simple_time_constr(model, t11, t12, t21, t22)

            #         break # only one line seg avoids
            #     break # only one line seg avoids
            # break # only one line seg avoids
        # break # only one agent avoids


class Node(object):
    """docstring for Node"""

    def __init__(self, op, deps=[], zs=[], info=[]):
        super(Node, self).__init__()
        self.op = op
        self.deps = deps
        self.zs = zs
        self.info = info


def clearSpecTree(spec):
    for dep in spec.deps:
        clearSpecTree(dep)
    spec.zs = []


def handleSpecTree(spec, PWL, bloat_factor, size):
    for dep in spec.deps:
        handleSpecTree(dep, PWL, bloat_factor, size)
    if len(spec.zs) == len(PWL) - 1:
        return
    elif len(spec.zs) > 0:
        raise ValueError('incomplete zs')
    if spec.op == 'mu':
        spec.zs = [mu(i, PWL, 0.1, spec.info['A'], spec.info['b'])
                   for i in range(len(PWL) - 1)]
    elif spec.op == 'negmu':
        spec.zs = [negmu(i, PWL, bloat_factor + size, spec.info['A'],
                         spec.info['b']) for i in range(len(PWL) - 1)]
    elif spec.op == 'and':
        spec.zs = [Conjunction([dep.zs[i] for dep in spec.deps])
                   for i in range(len(PWL) - 1)]
    elif spec.op == 'or':
        spec.zs = [Disjunction([dep.zs[i] for dep in spec.deps])
                   for i in range(len(PWL) - 1)]
    elif spec.op == 'U':
        spec.zs = [until(i, spec.info['int'][0], spec.info['int'][1],
                         spec.deps[0].zs, spec.deps[1].zs, PWL) for i in range(len(PWL) - 1)]
    elif spec.op == 'F':
        spec.zs = [eventually(i, spec.info['int'][0], spec.info['int']
        [1], spec.deps[0].zs, PWL) for i in range(len(PWL) - 1)]
    elif spec.op == 'BF':
        spec.zs = [bounded_eventually(i, spec.info['int'][0], spec.info['int'][1],
                                      spec.deps[0].zs, PWL, spec.info['tmax']) for i in
                   range(len(PWL) - 1)]
    elif spec.op == 'A':
        spec.zs = [always(i, spec.info['int'][0], spec.info['int']
        [1], spec.deps[0].zs, PWL) for i in range(len(PWL) - 1)]
    elif spec.op == 'N':
        spec.zs = [next(i, spec.deps[0].zs, PWL, spec.info['extra']) for i in range(len(PWL) - 1)]
    else:
        raise ValueError('wrong op code')


def gen_CDTree_constraints(model, root):
    if not hasattr(root, 'deps'):
        return [root, ]
    else:
        if len(root.constraints) > 0:
            # TODO: more check here
            return root.constraints
        dep_constraints = []
        for dep in root.deps:
            dep_constraints.append(gen_CDTree_constraints(model, dep))
        zs = []
        for dep_con in dep_constraints:
            if isinstance(root, Disjunction):
                z = model.addVar(vtype=GRB.BINARY)
                zs.append(z)
                dep_con = [con + M * (1 - z) for con in dep_con]
            root.constraints += dep_con
        if len(zs) > 0:
            root.constraints.append(sum(zs) - 1)
        model.update()
        return root.constraints


def add_CDTree_Constraints(model, root):
    constrs = gen_CDTree_constraints(model, root)
    for con in constrs:
        model.addConstr(con >= 0)


def plan(x0s, specs, bloat, limits=None, num_segs=None, tasks=None, vmax=3., MIPGap=1e-4,
         max_segs=None, tmax=None,
         hard_goals=None, size=0.11 * 4 / 2):
    if num_segs is None:
        min_segs = 1
        assert max_segs is not None
    else:
        min_segs = num_segs
        max_segs = num_segs

    for num_segs in range(min_segs, max_segs + 1):
        for spec in specs:
            clearSpecTree(spec)
        print('----------------------------')
        print('num_segs', num_segs)

        PWLs = []
        m = Model("xref")
        # m.setParam(GRB.Param.OutputFlag, 0)
        m.setParam(GRB.Param.IntFeasTol, IntFeasTol)
        m.setParam(GRB.Param.MIPGap, MIPGap)
        # m.setParam(GRB.Param.NonConvex, 2)
        # m.getEnv().set(GRB_IntParam_OutputFlag, 0)

        for idx_a in range(len(x0s)):
            x0 = x0s[idx_a]
            x0 = np.array(x0).reshape(-1).tolist()
            spec = specs[idx_a]

            dims = len(x0)

            PWL = []
            for i in range(num_segs + 1):
                PWL.append([m.addVars(dims, lb=-GRB.INFINITY), m.addVar()])
            PWLs.append(PWL)
            m.update()

            # the initial constriant (X0, T0)
            m.addConstrs(PWL[0][0][i] == x0[i] for i in range(dims))
            m.addConstr(PWL[0][1] == 0)

            if hard_goals is not None:
                goal = hard_goals[idx_a]
                m.addConstrs(PWL[-1][0][i] == goal[i] for i in range(dims))

            if limits is not None:
                add_space_constraints(m, [P[0] for P in PWL], limits)

            add_velocity_constraints(m, PWL, vmax=vmax)
            add_time_constraints(m, PWL, tmax)

            handleSpecTree(spec, PWL, bloat, size)
            add_CDTree_Constraints(m, spec.zs[0])

        if tasks is not None:
            for idx_agent in range(len(tasks)):
                for idx_task in range(len(tasks[idx_agent])):
                    handleSpecTree(tasks[idx_agent][idx_task],
                                   PWLs[idx_agent], bloat, size)

            conjunctions = []
            for idx_task in range(len(tasks[0])):
                disjunctions = [tasks[idx_agent][idx_task].zs[0]
                                for idx_agent in range(len(tasks))]
                conjunctions.append(Disjunction(disjunctions))
            z = Conjunction(conjunctions)
            add_CDTree_Constraints(m, z)

        add_mutual_clearance_constraints(m, PWLs, bloat)

        # obj = sum([L1Norm(m, _sub(PWL[i][0], PWL[i+1][0])) for PWL in PWLs for i in range(len(PWL)-1)])
        obj = sum([PWL[-1][1] for PWL in PWLs])
        m.setObjective(obj, GRB.MINIMIZE)

        m.write("test.lp")
        print('NumBinVars: %d' % m.getAttr('NumBinVars'))

        try:
            start = time.time()
            m.optimize()
            end = time.time()
            print('sovling it takes %.3f s' % (end - start))
            PWLs_output = []
            for PWL in PWLs:
                PWL_output = []
                for P in PWL:
                    PWL_output.append(
                        [[P[0][i].X for i in range(len(P[0]))], P[1].X])
                PWLs_output.append(PWL_output)
            m.dispose()
            return PWLs_output, end - start
        except Exception as e:
            m.dispose()
    return [None, None]


def transform_seq_time(seq_times, tmax):
    # transform seq time to fit tmax
    for _i in range(len(seq_times) - 1):
        seq_times[_i] = [(seq_times[_i][0] - 1) * (tmax / seq_times[-1][1]),
                         (seq_times[_i + 1][0] - 1) * (tmax / seq_times[-1][1])]
    if len(seq_times) > 1:
        seq_times[-1] = [seq_times[-2][1], tmax]
    return seq_times


def mip_plan(spec_id, rebuttal_mode=False):
    # Set velocity and time constraints
    tmax = 100.0
    vmax = 1.0
    num_segs = 20
    mip_gap = 1e-2
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    obs = []

    if not rebuttal_mode:
        # Include wall in non-rebuttal mode experiments
        # Boundary walls first
        walls = []

        wall_scaling_factor = 4 / 1.5

        walls.append(wall_scaling_factor *
                     np.array([-1.5, -1.5, -1.5, 1.5], dtype=np.float64))
        walls.append(wall_scaling_factor *
                     np.array([1.5, 1.5, -1.5, 1.5], dtype=np.float64))
        walls.append(wall_scaling_factor *
                     np.array([-1.5, 1.5, -1.5, -1.5], dtype=np.float64))
        walls.append(wall_scaling_factor *
                     np.array([-1.5, 1.5, 1.5, 1.5], dtype=np.float64))

        # Add inner walls, read hyperparameters from Spec info
        spec = PointSequence()
        x0 = spec.env.init_space.sample()
        # obs = []
        wall_half_width = spec.env.mj_env.config['walls_size']

        # Add outer walls

        for wall in walls:
            if wall[0] == wall[1]:
                wall[0] -= wall_half_width
                wall[1] += wall_half_width
            elif wall[2] == wall[3]:
                wall[2] -= wall_half_width
                wall[3] += wall_half_width
            else:
                raise ValueError("wrong shape for axis-aligned wall")
            wall *= np.array([-1, 1, -1, 1])
            obs.append((A, wall))

        for x, y in spec.env.mj_env.config['walls_locations']:
            wall = [x - wall_half_width, x + wall_half_width, y - wall_half_width,
                    y + wall_half_width]
            wall *= np.array([-1, 1, -1, 1])
            # print("adding wall", wall)
            obs.append((A, wall))

        # Avoid obstacles
        avoids = [Node('negmu', info={'A': A, 'b': b}) for A, b in obs]
        # avoids += [Node('negmu', info={'A':goals[j][0], 'b':goals[j][1]}) for j in range(4) if j is not i]
        avoid_obs = Node('and', deps=avoids)
        always_avoid_obs = Node('A', deps=[avoid_obs, ], info={'int': [0, tmax]})

    # Pick from set of specs here
    if spec_id == "seq":
        num_segs = 15
        # 0 : sequence
        if not rebuttal_mode:
            spec = PointSequence()
            rebuttal_spec = Seq()
            spec.close_r = rebuttal_spec.close_r
            spec.waypoints = spec.goals
            spec.seq_times = spec.seq_time
            x0 = spec.env.init_space.sample()
        else:
            spec = Seq()
            x0 = spec.env.sample_initial_state(1)

        # x0 = spec.env.sample_initial_state(1)
        goals = [(
            A,
            np.array(
                [
                    -(g[0] - spec.close_r),
                    (g[0] + spec.close_r),
                    -(g[1] - spec.close_r),
                    (g[1] + spec.close_r),
                ],
                dtype=np.float64,
            ),
        ) for g in spec.waypoints]

        plots = [[[goals[0], ], "y", ], [[goals[1], ], "r", ],
                 [[goals[2], ], "g", ], [[goals[3], ], "b", ], [obs, "k"], ]

        # Define predicates from goal boxes
        predicates = [Node("mu", info={"A": g[0], "b": g[1]}) for g in goals]
        # make phi list of eventuals
        seq_times = transform_seq_time(spec.seq_times, tmax)
        # After transform for tmax =15, seq_times are [[0.0, 3.75], [3.75, 7.5], [7.5, 11.25], [11.25, 15.0]]
        phis = [
            Node("F", deps=[p, ], info={"int": s},
                 ) for p, s in zip(predicates, seq_times)
        ]
        # reach all goals
        spec = Node("and", deps=phis)
    elif spec_id == "cover":
        # 1 : cover
        if not rebuttal_mode:
            spec = PointCover()
            rebuttal_spec = Cover()
            spec.close_r = rebuttal_spec.close_r
            spec.waypoints = spec.goals
            x0 = random.choice([[3, -3], [-3, -3], [3, 3]])
        else:
            spec = Cover()
            x0 = spec.env.sample_initial_state(1)
        # get goal boxes
        goals = [(
            A,
            np.array(
                [
                    -(g[0] - spec.close_r),
                    (g[0] + spec.close_r),
                    -(g[1] - spec.close_r),
                    (g[1] + spec.close_r),
                ],
                dtype=np.float64,
            ),
        ) for g in spec.waypoints]

        plots = [[[goals[0], ], "y", ], [[goals[1], ], "r", ],
                 [[goals[2], ], "g", ], [[goals[3], ], "b", ], [obs, "k"], ]

        # Define predicates from goal boxes
        predicates = [Node("mu", info={"A": g[0], "b": g[1]}) for g in goals]
        # make phi list of eventuals
        phis = [
            Node("F", deps=[p, ], info={"int": [0, tmax]},
                 ) for p in predicates
        ]
        # reach all goals
        spec = Node("and", deps=phis)
    elif spec_id == "branch":
        # 2 : branch
        if not rebuttal_mode:
            spec = PointBranch()
            rebuttal_spec = Branch()
            spec.close_r = rebuttal_spec.close_r
        else:
            spec = Branch()
            tmax = 10  # rm this makes plan very inefficient
            x0 = spec.env.sample_initial_state(1)
        # get goal boxes
        branches = [[(
            A,
            np.array(
                [
                    -(g[0] - spec.close_r),
                    (g[0] + spec.close_r),
                    -(g[1] - spec.close_r),
                    (g[1] + spec.close_r),
                ],
                dtype=np.float64,
            ),
        ) for g in b] for b in spec.branches]

        plots = [[[branches[0][0], ], "y", ], [[branches[0][1], ], "r", ],
                 [[branches[1][0], ], "g", ], [[branches[1][1], ], "b", ],
                 [obs, "k"], ]

        # Define predicates from goal boxes
        predicates = [[Node("mu", info={"A": g[0], "b": g[1]})
                       for g in b] for b in branches]
        # make phi list of eventuals
        seq_times = transform_seq_time([[1, 2], [3, 4]], tmax)
        # After transform for tmax =15, seq_times are [[0.0, 7.5], [7.5, 15.0]]
        phis = [[
            Node("F", deps=[p, ], info={"int": s},
                 ) for p, s in zip(pred_b, seq_times)
        ] for pred_b in predicates]
        # reach all goals in a branch
        spec = Node("or", deps=[Node("and", deps=branch_seq)
                                for branch_seq in phis])

    elif spec_id == "loop":
        # 3 : loop
        if not rebuttal_mode:
            spec = PointLoop()
            rebuttal_spec = Loop()
            spec.close_r = rebuttal_spec.close_r
            num_loops = 5
        else:
            spec = Loop()
            spec.waypoints = spec.loop_wps
            num_loops = 10
            x0 = spec.env.sample_initial_state(1)
        # get goal boxes
        goals = [(
            A,
            np.array(
                [
                    -(g[0] - spec.close_r),
                    (g[0] + spec.close_r),
                    -(g[1] - spec.close_r),
                    (g[1] + spec.close_r),
                ],
                dtype=np.float64,
            ),
        ) for g in spec.waypoints]
        plots = list(zip(goals, "yrgb"[:len(goals)])) + [[obs, "k"], ]

        # Define predicates from goal boxes
        predicates = [Node("mu", info={"A": g[0], "b": g[1]}) for g in goals]
        notpredicates = [Node("negmu", info={"A": g[0], "b": g[1]}) for g in goals]
        # make phi list of eventually reach a goal
        phis = [
            Node("F", deps=[p, ], info={"int": [0, tmax]},
                 ) for p in predicates
        ]

        def get_next_pred_list(pred, nump=3):
            return [Node("N", deps=[pred], info={'extra': _i}) for _i in range(1, nump + 1)]

        def get_next_pred(pred, nump=3):
            return [Node("N", deps=[pred], info={'extra': nump})]

        # Paper version of loop (not working as needed)
        # Try paper spec
        # reach all goals
        # either reach a goal
        # lh_spec = Node("or", deps=predicates)
        # # # orloop condition
        # rh_spec_part = [Node("or",deps=[notp,] + get_next_pred(notp, 5) ) for notp in notpredicates]
        # rh_spec = Node("and", deps=rh_spec_part)
        # loop_invar_spec = Node('A', deps=[Node("and",deps=[lh_spec,rh_spec]),], info={'int':[1, 10]})
        # spec =Node("and",deps=[loop_invar_spec] + [phis[2]])
        # # reach all goals
        # spec = Node("and", deps=[spec] + [always_avoid_obs])

        # Loop as sequence
        spec = PointSequence()

        phi_list = []

        num_segs = 50
        mip_gap = 0.5

        seq_times_base = transform_seq_time(spec.seq_time, tmax / num_loops)
        for i in range(num_loops):
            seq_times = seq_times_base.copy()
            seq_times = np.array(seq_times) + (tmax / num_loops) * i
            # After transform for tmax =15, seq_times are [[0.0, 3.75], [3.75, 7.5], [7.5, 11.25], [11.25, 15.0]]
            phis = [
                Node("F", deps=[p, ], info={"int": s},
                     ) for p, s in zip(predicates, seq_times)
            ]
            phi_list += phis
        # reach all goals
        spec = Node("and", deps=phi_list)
    elif spec_id == "signal":
        # 4 : signal
        num_segs = 50
        mip_gap = 0.5
        if not rebuttal_mode:
            spec = PointSignal(enable_gui=False)
            rebuttal_spec = Signal()
            spec.close_r = rebuttal_spec.close_r
            # 20 time steps, 5 loops + 1 for final goal
            num_loops = 6
        else:
            spec = Signal()
            spec.loop_waypoints = tuple(spec.loop_wps)
            # 20 time steps, 10 loops + 1 for final goal
            num_loops = 11
            x0 = spec.env.sample_initial_state(1)
        # get goal boxes
        goals = [(
            A,
            np.array(
                [
                    -(g[0] - spec.close_r),
                    (g[0] + spec.close_r),
                    -(g[1] - spec.close_r),
                    (g[1] + spec.close_r),
                ],
                dtype=np.float64,
            ),
        ) for g in spec.loop_waypoints + (spec.final_goal,)]

        plots = list(zip(goals, "yrgbm"[:len(goals)])) + [[obs, "k"], ]

        # Define predicates from goal boxes
        predicates = [Node("mu", info={"A": g[0], "b": g[1]})
                      for g in goals[:-1]]
        final_goal_predicate = Node(
            "mu", info={"A": goals[-1][0], "b": goals[-1][1]})
        # make phi list of eventually reach a goal
        phis = [
            Node("F", deps=[p, ], info={"int": [0, tmax]},
                 ) for p in predicates
        ]

        # Signal as Sequence

        seq_spec = PointSequence()

        phi_list = []

        seq_times_base = transform_seq_time(seq_spec.seq_time, tmax / num_loops)
        for i in range(num_loops - 1):
            seq_times = seq_times_base.copy()
            seq_times = np.array(seq_times) + (tmax / num_loops) * i
            # After transform for tmax =15, seq_times are [[0.0, 3.75], [3.75, 7.5], [7.5, 11.25], [11.25, 15.0]]
            phis = [
                Node("F", deps=[p, ], info={"int": s},
                     ) for p, s in zip(predicates, seq_times)
            ]
            phi_list += phis
        # reach all goals then final goal
        phi_list += [
            Node("F", deps=[final_goal_predicate, ], info={"int": [seq_times[-1][1], tmax]},
                 )
        ]
        spec = Node("and", deps=phi_list)

    else:
        raise NotImplementedError

    if not rebuttal_mode and not spec_id == "signal":
        # reach all goals and avoid obstacles too in environments with walls
        spec = Node("and", deps=[spec] + [always_avoid_obs])

    x0s = [x0, ]
    specs = [spec, ]
    goals = None  # [goal, ]
    PWL, planning_time = plan(
        x0s,
        specs,
        bloat=0.05,
        size=0.4,  # based on mujoco robot keepout
        num_segs=num_segs,
        MIPGap=mip_gap,
        tmax=tmax,
        vmax=vmax,
        hard_goals=goals,
    )

    return x0s, plots, PWL, planning_time