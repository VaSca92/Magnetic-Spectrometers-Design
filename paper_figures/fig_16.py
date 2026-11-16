import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from time import time


# Input --------------------------------------------------------------------------------------------
scattering = [1e0, 1e2, 1e4, 1e6]
min_length = 0.0
max_length = 2.0
plot_points = 200
filling = 0.8

# Optimizator configuration ------------------------------------------------------------------------
opt = {"ftol": 1e-10, "eps": 1e-9, "maxiter": 1000}
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
    m = c @ x - np.cumsum(c*x); # eq .(9)
    # normalized scattering error
    s = (f*x) @ ((c*x)**2 + 3* c * x * m + 3 * m**2)
    
    # normalized resolution
    r = np.sqrt (d + p*s) / np.abs(b) # eq .(12)-(13)
    return r / np.sqrt(p + 16) # eq .(17)

# Define initial point for optimization ------------------------------------------------------------
def generate_x0(scattering):
    distance = 1-(32/scattering)**(1/3)
    c = np.array([+1, 0, -1])
    x = np.array([(1-distance)/2, distance, (1-distance)/2])
    return np.concatenate([c,x])

# Prepare for optimization -------------------------------------------------------------------------
lengths = np.linspace(min_length, max_length, plot_points)
norm_res = np.zeros(lengths.size)

# Figure 16: normalized resolution vs calorimeter length
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
        constraintN = lambda x: x[:n] @ (x[n:] + np.array([l,0,0]))
        ncon = NonlinearConstraint(constraintN, 0, 0)
        # linear constraint
        constraintL = np.concatenate([np.zeros(n), np.ones(n)])
        lcon = LinearConstraint(constraintL, 1, 1)
        # max and min bound for variables
        boundmin = np.append(1, np.concatenate([-np.inf*np.ones(n-1), np.zeros(n)]))
        boundmax = np.append(1, np.concatenate([+np.inf*np.ones(n-1), np. ones(n)]))    
        bounds = Bounds(boundmin, boundmax)

        # Optimization -----------------------------------------------------------------------------
        fun = lambda x: resolution(x[:n], x[n:], s, l)
        sol = minimize(fun, x0, method="slsqp", options=opt, bounds=bounds, constraints=[lcon,ncon])
        norm_res[i] = sol.fun
        x0 = sol.x + shaking*np.random.randn(sol.x.size)
    tend = time()

    plt.plot(lengths, norm_res, label=f"Scattering parameter = {s:.2e}")
    
    print(f"plot with scattering = {s:.2e} computed")
    print(f"elapsed time: {tend-tstart:.3f}")

# Plotting results ---------------------------------------------------------------------------------
eq_len = filling * lengths
plt.plot(lengths, 1/(1+eq_len)**2, '--', label=r"scattering parameter = 0 (eq. 42)")
plt.plot(lengths, np.sqrt(3/4/(1+eq_len)), '--', label=r"scattering parameter -> $\infty$ (eq. 43)")
plt.legend()
plt.xlabel(r"magnet length $L$")
plt.ylabel(r"normalized resolution $\psi$")
plt.grid()
plt.axis([min_length, max_length, 0.2, 1])

plt.show()


