from fairrecs.solver import Solver
import cvxpy as cp


class DPSolver(Solver):
    def __init__(self, u):
        super(u, self).__init__()

    def get_fair_constraint(self):
        u = self.u
        v = self.v
        P = self.P
        # TODO: Define f here
        fair_constraint = cp.matmul(cp.matmul(f.transpose(), P), v)
        self.constraints.append(fair_constraint)
