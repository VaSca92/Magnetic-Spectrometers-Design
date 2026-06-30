import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from time import time

# Input --------------------------------------------------------------------------------------------
scattering = [1e1, 1e2, 1e3, 1e4]
min_length = 0.5
max_length = 1.0
plot_points = 200

# Optimizator configuration ------------------------------------------------------------------------
opt = {"ftol": 1e-9, "eps": 1e-9, "maxiter": 1000}
shaking = 1e-5; np.random.seed(0)

# Define resolution function -----------------------------------------------------------------------
f = np.array([0, 1, 1, 1, 0]) # filling factor
def resolution (c, x, p, l):
    c = np.hstack([c[0], c, c[-1]])
    x = np.hstack([l[0], x, l[-1]])
    
    # normalized bending power
    b = c @ (x * (2* np.cumsum (f*x) - f*x))
    # normalized tracking error
    d = np.sum( np.diff(c ,1 ,0 ,0 ,0)**2)
    m = c @ x - np.cumsum(c*x); # eq .(9)
    # normalized scattering error
    s = (f*x) @ ((c*x)**2 + 3* c * x * m + 3 * m**2)
    
    # normalized resolution
    r = np.sqrt (d + p*s) / np.abs(b) # eq. (12)-(13)
    return r / np.sqrt(p + 16) # eq. (17)

# Define initial point for optimization ------------------------------------------------------------
def generate_x0(scattering):
    distance = 1-(32/scattering)**(1/3)
    c = np.array([+1, 0, -1])
    x = np.array([(1-distance)/2, distance, (1-distance)/2])
    return np.concatenate([c,x])

# Prepare for optimization -------------------------------------------------------------------------
lengths = np.linspace(max_length, min_length, plot_points)
norm_res = np.zeros(lengths.size)

# Figure 13: normalized resolution vs scattering parameter
plt.figure()

print("Starting optimization for different values of scattering parameter")
# Loop over scattering values
for s in scattering:

    # Define initial condition
    x0 = generate_x0(s)
    n = x0.size // 2

    tstart = time()
    for i, l in enumerate(lengths):

        # Define constraints -----------------------------------------------------------------------
        # nonlinear constraint
        constraintN = lambda x: x[:n] @ (x[n:] + np.stack([1-l,0,1-l])/2)
        ncon = NonlinearConstraint(constraintN, 0, 0)
        # linear constraint
        constraintL = np.concatenate([np.zeros(n), np.ones(n)])
        lcon = LinearConstraint(constraintL, l, l)
        # max and min bound for variables
        boundmin = np.concatenate([-np.inf*np.ones(n), np.zeros(n)])
        boundmax = np.concatenate([+np.inf*np.ones(n), np.ones(n)*l])    
        bounds = Bounds(boundmin, boundmax)

        # Optimization -----------------------------------------------------------------------------
        fun = lambda x: resolution(x[:n], x[n:], s, (1 - np.stack([l,l]))/2)
        sol = minimize(fun, x0, method="slsqp", options=opt, bounds=bounds, constraints=[lcon,ncon])
        norm_res[i] = sol.fun
        x0 = sol.x + shaking*np.random.randn(sol.x.size)

    tend = time()

    plt.plot(lengths, norm_res, label=f"scattering parameter = {s:.2e}")

    print(f"plot with scattering = {s:.2e} computed")
    print(f"elapsed time: {tend-tstart:.3f}")


# Plotting results ---------------------------------------------------------------------------------
plt.plot(lengths, 1/(2-lengths)/lengths, '--', label=r"scattering parameter = 0 (eq. 37)")
plt.plot(lengths, np.sqrt(3/4/lengths), '--', label=r"scattering parameter -> $\infty$ (eq. 38)")
plt.legend()
plt.xlabel(r"magnet length $L$")
plt.ylabel(r"normalized resolution $\psi$")
plt.grid()
plt.axis([min_length, max_length, 0.85, 1.35])

plt.show()
