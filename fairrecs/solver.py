import numpy as np
import cvxpy as cp


class Solver(object):
    def __init__(self, u):
        self.u = u
        self.v = np.array([1.0 / (np.log(2 + i)) for i, _ in enumerate(u)])
        self.P = cp.Variable((len(u), len(u)))
        self.I = np.ones((len(u),))
        self.constraints = [cp.matmul(self.I.transpose(), self.P) == self.I.transpose(),
                       cp.matmul(self.P, self.I) == self.I,
                       0 <= self.P, self.P <= 1
                       ]

    def solve(self):
        u = self.u
        v = self.v
        P = self.P
        constraints = self.constraints
        objective = cp.Maximize(cp.matmul(cp.matmul(u, P), v))
        self.get_fair_constraint()

        prob = cp.Problem(objective, constraints)

        result = prob.solve(verbose=False, solver=cp.SCS)
        return P.value

    def get_fair_constraint(self):
        return

    def expected_utility(self):
        u = self.u
        v = self.v
        P = self.P

        return cp.matmul(cp.matmul(u, P.value), v)