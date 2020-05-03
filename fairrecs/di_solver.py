from fairrecs.solver import Solver
import numpy as np
import cvxpy as cp


class DISolver(Solver):
    def __init__(self, u, g, alpha=None):
        super(u, self).__init__()
        self.alpha = alpha

        # Create indicator vectors for the two groups
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
        u = self.u
        v = self.v
        P = self.P
        G1 = self.G1
        G2 = self.G2

        # Get the average utility of the two groups:
        UG1 = np.sum(u[np.where(G1 == 1)])
        UG2 = np.sum(u[np.where(G2 == 1)])

        # need to check the definition of f (are we missing 1/|G1| and 1/|G2|?)
        # if so, then we need to change dt_solver

        f = (self.G1 / UG1 - self.G2 / UG2)*u

        if self.alpha is None:
            fair_constraint = (cp.matmul(cp.matmul(f.transpose(), P), v) == 0)
            self.constraints.append(fair_constraint)
        else:
            fair_constraint1 = (cp.matmul(cp.matmul(f.transpose(), P), v) <= self.alpha)
            fair_constraint2 = (-self.alpha <= cp.matmul(cp.matmul(f.transpose(), P), v))
            self.constraints.append(fair_constraint1)
            self.constraints.append(fair_constraint2)

