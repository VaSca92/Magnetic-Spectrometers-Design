import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from time import time

# Input --------------------------------------------------------------------------------------------
min_scattering = 1e0
max_scattering = 1e5
plot_points = 100

# Optimizator configuration ------------------------------------------------------------------------
opt = {"ftol": 1e-12, "eps": 1e-8, "maxiter": 1000}
shaking = 1e-5; np.random.seed(0)

# Define initial condition
c = np.array([+1, 0, -1])
x = np.array([.5, 0, .5])
x0 = np.concatenate([c,x])


# Define resolution function -----------------------------------------------------------------------
def resolution (c, x, p): # input: coefficients, distances, scattering parameter
    # normalized bending power
    b = c @ (x * (2* np.cumsum (x) - x))
    # normalized tracking error
    d = np.sum( np.diff(c ,1 ,0 ,0 ,0)**2)
    m = c @ x - np.cumsum(c*x); # eq .(9)
    # normalized scattering error
    s = x @ ((c*x)**2 + 3* c * x * m + 3 * m**2)
    
    # normalized resolution
    r = np.sqrt (d + p*s) / np.abs(b) # eq .(12)-(13)
    return r / np.sqrt(p + 16) # eq .(17)


# Define constraints -------------------------------------------------------------------------------
n = x0.size // 2

# nonlinear constraint
constraintN = lambda x: x[:n] @ x[n:]
ncon = NonlinearConstraint(constraintN, 0, 0)
# linear constraint
constraintL = np.concatenate([np.zeros(n), np.ones(n)])
lcon = LinearConstraint(constraintL, 1, 1)
# max and min bound for variables
boundmin = np.concatenate([-np.inf*np.ones(n), np.zeros(n)])
boundmax = np.concatenate([+np.inf*np.ones(n), np. ones(n)])
bounds = Bounds(boundmin, boundmax)

# Optimization -------------------------------------------------------------------------------------
scattering = np.logspace(np.log10(min_scattering), np.log10(max_scattering), plot_points)
distance = np.zeros(scattering.size)
norm_res = np.zeros(scattering.size)

tstart = time()
for i, s in enumerate(scattering):
    obj_fun = lambda x: resolution(x[:n], x[n:], s)
    sol = minimize(obj_fun, x0, method="slsqp", options=opt, bounds=bounds, constraints=[lcon,ncon])
    distance[i] = sol.x[-2]; norm_res[i] = sol.fun
    x0 = sol.x + shaking*np.random.randn(6)
tend = time()
print(f"elapsed time: {tend-tstart:.3f}")

# Plotting results ---------------------------------------------------------------------------------
# Figure 6: distance between central trackers vs scattering parameter
plt.figure()
plt.semilogx(scattering, distance)
plt.grid()
plt.axis([min_scattering, max_scattering, -0.01, 1.0])
plt.xlabel(r"scattering parameter $\xi$")
plt.ylabel(r"distance between trackers $\chi_2$")

# Figure 7: normalized resolution vs scattering parameter
plt.figure()
plt.semilogx(scattering, norm_res)
plt.grid()
plt.axis([min_scattering, max_scattering, 0.88, 1.01])
plt.xlabel(r"scattering parameter $\xi$")
plt.ylabel(r"normalized resolution $\psi$")

plt.show()