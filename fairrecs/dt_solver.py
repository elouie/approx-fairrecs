from fairrecs.solver import Solver
import numpy as np
import cvxpy as cp


class DTSolver(Solver):
    def __init__(self, u, g):
        super().__init__(u)

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

        f = self.G1 / UG1 - self.G2 / UG2
        fair_constraint = (cp.matmul(cp.matmul(f.transpose(), P), v) == 0)
        self.constraints.append(fair_constraint)
