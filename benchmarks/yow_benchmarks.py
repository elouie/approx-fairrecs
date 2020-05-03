import cvxpy as cp


def cost_of_fairness(u, P, P_fair, v):
    return cp.matmul(cp.matmul(u, (P - P_fair)), v)
