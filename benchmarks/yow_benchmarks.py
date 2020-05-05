import numpy as np


def cost_of_fairness(u, P, P_fair, v):
    return np.dot(np.dot(u, (P - P_fair)), v)
