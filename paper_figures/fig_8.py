import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from time import time

# Input --------------------------------------------------------------------------------------------
min_scattering = 1e0
max_scattering = 1e5
min_trackers = 3
max_trackers = 8
plot_points = 200

# Optimizator configuration ------------------------------------------------------------------------
opt = {"ftol": 1e-9, "eps": 1e-6, "maxiter": 1000}
shaking = 1e-5; np.random.seed(0)

# Define resolution function -----------------------------------------------------------------------
def resolution (c, x, p):
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

# Define initial point for optimization ------------------------------------------------------------
def generate_x0(trackers):
    n = (trackers + 1)// 4
    m = trackers - 2*n 
    c = np.cumsum(np.concatenate([np.tile(1/n,n), np.tile(-2/m,m), np.tile(1/n,n)]))[:-1]
    x = np.zeros(trackers-1); x[n-1] = 0.5; x[trackers-1-n] = 0.5        
    return np.concatenate([c, x])

# Prepare for optimization -------------------------------------------------------------------------
scattering = np.logspace(np.log10(min_scattering), np.log10(max_scattering), plot_points)
norm_res = np.zeros(scattering.size)

# Figure 8: normalized resolution vs scattering parameter
plt.figure()
if min_trackers <= 3: # 3 trackers computed analytically
    plt.plot(scattering, np.sqrt((scattering+24) / (scattering+16)), label = "3 Trackers")
    min_trackers = 4 

print("Starting optimization for different number of trackers")
# Loop over number of trackers
for tracker_num in range(min_trackers, max_trackers + 1):

    x0 = generate_x0(tracker_num)
    n = tracker_num - 1

    # Define constraints ---------------------------------------------------------------------------
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

    # Optimization ---------------------------------------------------------------------------------
    tstart = time()
    for i, s in enumerate(scattering):
        fun = lambda x: resolution(x[:n], x[n:], s)
        sol = minimize(fun, x0, method="slsqp", options=opt, bounds=bounds, constraints=[lcon,ncon])
        norm_res[i] = sol.fun
        x0 = sol.x + shaking*np.random.randn(sol.x.size)
    tend = time()

    plt.semilogx(scattering, norm_res, label=f"{tracker_num} Trackers")

    print(f"plot for N = {tracker_num - 1} computed")
    print(f"elapsed time: {tend-tstart:.3f}")
    
# Plotting results ---------------------------------------------------------------------------------
plt.legend()
plt.xlabel(r"scattering parameter $\xi$")
plt.ylabel(r"normalized resolution $\psi$")
plt.grid()
plt.axis([min_scattering, max_scattering, 0.7, 1.25])

plt.show()
