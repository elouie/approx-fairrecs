import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, u):
        v = np.array([1.0/(np.log(2 + i)) for i, _ in enumerate(u)])
        P = cp.Variable((len(u), len(u)))
        objective = cp.Maximize(cp.matmul(cp.matmul(u, P), v))
        constraints = [cp.matmul(np.ones((1,len(u))), P) == np.ones((1,len(u))),
                       cp.matmul(P, np.ones((len(u),))) == np.ones((len(u),)),
                       0 <= P, P <= 1,
                       self.get_fair_constraint()
                      ]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        # result = prob.solve(verbose=True, max_iters=1000)
        result = prob.solve(verbose=True, solver=cp.SCS)
        return P.value

    def get_fair_constraint(self):
        return None
