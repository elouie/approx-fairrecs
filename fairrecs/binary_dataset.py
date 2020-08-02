from scipy.stats import truncnorm
from numpy import hstack, zeros, ones


class GroupSettings:
    def __init__(self, size, mu, sigma):
        self.size = size
        self.mu = mu
        self.sigma = sigma


class BinaryDataset:
    """
        args:
            a (GroupSettings)
            b (GroupSettings)
    """
    def __init__(self, a, b):
        lower, upper = 0, 100
        mu, sigma = 20, 20
        A = truncnorm((lower - a.mu) / a.sigma, (upper - a.mu) / a.sigma, loc=a.mu, scale=a.sigma)
        B = truncnorm((lower - b.mu) / b.sigma, (upper -b. mu) / b.sigma, loc=b.mu, scale=b.sigma)
        self.u = hstack([A.rvs(a.size), B.rvs(b.size)])
        self.G1 = hstack([ones(a.size), zeros(b.size)])
        self.G2 = hstack([zeros(a.size), ones(b.size)])
