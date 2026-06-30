import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from time import time

# Input --------------------------------------------------------------------------------------------
min_scattering = 1e0
max_scattering = 1e4
lengths = [0.1, 0.5, 1.0, 1.5, 2.0]
plot_points = 200
filling = 0.8

# Optimizator configuration ------------------------------------------------------------------------
opt = {"ftol": 1e-8, "eps": 1e-8, "maxiter": 1000}
shaking = 1e-5; np.random.seed(0)

# Define resolution function -----------------------------------------------------------------------
f = np.array([filling, 1, 1, 1])
def resolution (c, x, p, l):
    c = np.hstack([c[0], c])
    x = np.hstack([l, x])
    
    # normalized bending power
    b = c @ (x * (2* np.cumsum (f*x) - f*x))
    # normalized tracking error
    d = np.sum( np.diff(c ,1 ,0 ,0 ,0)**2)
    
    # normalized scattering error
    s = f[0] * c[0]**2 * l**3
    
    # normalized resolution
    return np.sqrt (d + p*s) / np.abs(b) / 4 # eq .(12)-(13)

# Define initial point for optimization ------------------------------------------------------------
def generate_x0(length):
    distance = np.min((1-length)/2, 0)
    c = np.array([1-distance, 0, -distance-length])
    x = np.array([distance, 0, 1-distance])
    c /= c[0]
    return np.concatenate([c,x])

# Figure 17: normalized resolution vs scattering parameter
plt.figure()

# Prepare for optimization -------------------------------------------------------------------------
scattering = np.logspace(np.log10(min_scattering), np.log10(max_scattering), plot_points)
norm_res = np.zeros(scattering.size)

print("Starting optimization for different calorimeter lengths")
#loop over calorimtere lengths
for l in lengths:
    
    x0 = generate_x0(l)
    n = x0.size // 2
    
    tstart = time()
    for i, s in enumerate(scattering):
        # Define constraints -----------------------------------------------------------------------
        # nonlinear constraint
        constraintN = lambda x: x[:n] @ (x[n:] + np.array([l,0,0]))
        ncon = NonlinearConstraint(constraintN, 0, 0)
        # linear constraint
        constraintL = np.concatenate([np.zeros(n), np.ones(n)])
        lcon = LinearConstraint(constraintL, 1, 1)
        # bounds (fix c1 = 1)
        boundmin = np.append(1, np.concatenate([-np.inf*np.ones(n-1), np.zeros(n)]))
        boundmax = np.append(1, np.concatenate([+np.inf*np.ones(n-1), np. ones(n) ]))    
        bounds = Bounds(boundmin, boundmax)

        # Optimization -----------------------------------------------------------------------------
        fun = lambda x: resolution(x[:n], x[n:], s, l)
        sol = minimize(fun, x0, method="slsqp", options=opt, bounds=bounds, constraints=[lcon,ncon])
        norm_res[i] = sol.fun
        x0 = sol.x + shaking*np.random.randn(sol.x.size)
    tend = time()

    plt.semilogx(scattering, norm_res, label=f"calorimeter length = {l:.1f}")
    
    print(f"calorimeter length {l:.1f} computed")
    print(f"elapsed time: {tend-tstart:.3f}")

plt.legend()
plt.xlabel(r"scattering parameter $\xi$")
plt.ylabel(r"normalized resolution $\psi$")
plt.grid()
plt.axis([min_scattering, max_scattering, 0.2, 1.25])

plt.show()