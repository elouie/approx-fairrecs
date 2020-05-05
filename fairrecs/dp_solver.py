from fairrecs.solver import Solver
import numpy as np
import cvxpy as cp


class DPSolver(Solver):
    def __init__(self, u, g):
        super().__init__(u)
        self.alpha = None
        uniq_ids = np.unique(g)
        g1 = np.where(g == uniq_ids[0])
        G1 = np.zeros(g.shape)
        G1[g1] = 1
        self.G1 = G1
        g2 = np.where(g == uniq_ids[1])
        G2 = np.zeros(g.shape)
        G2[g2] = 1
        self.G2 = G2

    def get_fair_constraint(self):
        v = self.v
        P = self.P
        f_G1 = self.G1 / abs(np.sum(self.G1))
        f_G2 = self.G2 / abs(np.sum(self.G2))
        if self.alpha is None:
            fair_constraint = (cp.matmul(cp.matmul((f_G1 - f_G2).transpose(), P), v) == 0)
            self.all_constraints.append(fair_constraint)
        else:
            fair_constraint1 = (cp.matmul(cp.matmul((self.alpha * f_G1 - f_G2).transpose(), P), v) <= 0)
            fair_constraint2 = (cp.matmul(cp.matmul((self.alpha * f_G2 - f_G1).transpose(), P), v) <= 0)
            self.all_constraints.append(fair_constraint1)
            self.all_constraints.append(fair_constraint2)
