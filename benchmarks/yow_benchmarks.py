import numpy as np
from fairrecs.preprocessor import preprocess_yow
from fairrecs.solver import Solver
from fairrecs.dp_solver import DPSolver
from fairrecs.dt_solver import DTSolver
from fairrecs.di_solver import DISolver
from fairrecs.utils.cof import cost_of_fairness as cof
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import seaborn as sns

# Configuration
sns.set_palette('colorblind')
dataset = preprocess_yow('../datasets/yow_userstudy_raw.csv')

solver = Solver(dataset['relevant'].to_numpy())
P = solver.solve()

solver_dp = DPSolver(dataset['relevant'].to_numpy(), dataset['RSS_ID'].to_numpy())
P_dp = solver_dp.solve()

solver_dt = DTSolver(dataset['relevant'].to_numpy(), dataset['RSS_ID'].to_numpy())
P_dt = solver_dt.solve()

solver_di = DISolver(dataset['relevant'].to_numpy(), dataset['RSS_ID'].to_numpy())
P_di = solver_di.solve()

u = dataset['relevant'].to_numpy()
v = np.array([1.0 / (np.log(2 + i)) for i, _ in enumerate(u)])

print("Expected utility: ", solver.expected_utility())
print("Demographic parity expected utility: ", solver_dp.expected_utility())
print("Cost of fairness: ", cof(u, P, P_dp, v))
print("Disparate treatment expected utility: ", solver_dt.expected_utility())
print("Cost of fairness: ", cof(u, P, P_dt, v))
print("Expected utility: ", solver.expected_utility())
print("Disparate impact expected utility: ", solver_di.expected_utility())
solver_dpa = DPSolver(dataset['relevant'].to_numpy(), dataset['RSS_ID'].to_numpy())
solver_dta = DTSolver(dataset['relevant'].to_numpy(), dataset['RSS_ID'].to_numpy())
solver_dia = DISolver(dataset['relevant'].to_numpy(), dataset['RSS_ID'].to_numpy())
count = 6
X = 1 - np.logspace(3,1,count)/2000
Y = np.zeros([count, 3])
C = np.zeros([count, 3])
for i, alpha in enumerate(X):
    if i % 3 == 1:
        print("Round ", i)
    solver_dpa.alpha = alpha
    solver_dta.alpha = alpha
    solver_dia.alpha = alpha
    P_dpa = solver_dpa.solve()
    P_dta = solver_dta.solve()
    P_dia = solver_dia.solve()
    Y[i, 0] = solver_dpa.expected_utility()
    Y[i, 1] = solver_dta.expected_utility()
    Y[i, 2] = solver_dia.expected_utility()
    C[i, 0] = cof(u, P, P_dpa, v)
    C[i, 1] = cof(u, P, P_dta, v)
    C[i, 2] = cof(u, P, P_dia, v)

plt.figure(num=None, figsize=(8, 6), dpi=300)
plot(X, C[:,0], label="Demographic parity")
plot(X, C[:,1], label="Disparate treatment")
plot(X, C[:,2], label="Disparate impact")
plt.xlabel('Alpha (Percent disparity)')
plt.ylabel('Cost of fairness')
plt.legend()
plt.show()

plt.figure(num=None, figsize=(8, 6), dpi=300)
plt.ticklabel_format(useOffset=False)
plot(X, Y[:,0], label="Demographic parity")
plot(X, Y[:,1], label="Disparate treatment")
plot(X, Y[:,2], label="Disparate impact")
plt.xlabel('Alpha (Percent disparity)')
plt.ylabel('Average utility')
plt.legend()
plt.show()
