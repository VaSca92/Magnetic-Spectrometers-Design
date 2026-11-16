import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from time import time

# Input --------------------------------------------------------------------------------------------
min_scattering = 1e0
max_scattering = 1e5
min_trackers = 3
max_trackers = 16
plot_points = 200

# Optimizator configuration ------------------------------------------------------------------------
opt = {"ftol": 1e-9, "eps": 1e-6, "maxiter": 1000}
shaking = 1e-5; np.random.seed(0)

# Define resolution function -----------------------------------------------------------------------
def resolution_no_scattering (c, x):
    # normalized bending power
    b = c @ (x * (2* np.cumsum (x) - x))
    # normalized tracking error
    d = np.sum( np.diff(c ,1 ,0 ,0 ,0)**2)
    
    # normalized resolution
    return np.sqrt (d) / np.abs(b) / 4 # eq .(12)-(13)
    

# Define initial point for optimization ------------------------------------------------------------
def generate_x0(trackers):
    n = (trackers + 1)// 4
    m = trackers - 2*n 
    c = np.cumsum(np.concatenate([np.tile(1/n,n), np.tile(-2/m,m), np.tile(1/n,n)]))[:-1]
    x = np.zeros(trackers-1); x[n-1] = 0.5; x[trackers-1-n] = 0.5        
    return np.concatenate([c, x])

# Prepare for optimization -------------------------------------------------------------------------
scattering = np.logspace(np.log10(min_scattering), np.log10(max_scattering), plot_points)
norm_res = np.zeros(max_trackers - min_trackers + 1)

# Figure 8: normalized resolution vs scattering parameter
plt.figure()

print("Starting optimization for different number of trackers")
# Loop over number of trackers
for i, tracker_num in enumerate(range(min_trackers, max_trackers + 1)):

    if tracker_num == 3:
        norm_res[i] = np.sqrt(3/2)
        continue

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
    
    fun = lambda x: resolution_no_scattering(x[:n], x[n:])
    sol = minimize(fun, x0, method="slsqp", options=opt, bounds=bounds, constraints=[lcon,ncon])
    norm_res[i] = sol.fun

    tend = time()

tracker_num = np.arange(min_trackers, max_trackers + 1) 
plt.plot(tracker_num, norm_res, label="Numerical Optimization", marker='o', linestyle="none")

# plot formula 33 vs numerical 
tracker_num = np.linspace(min_trackers, max_trackers, 100)
plt.plot( tracker_num, 2/np.sqrt(tracker_num), label="Analytical Approximation")

    
# Plotting results ---------------------------------------------------------------------------------
plt.legend()
plt.xlabel(r"Number of trackers N+1")
plt.ylabel(r"normalized resolution $\psi$")
plt.grid()

plt.xlim(min_trackers, max_trackers)
plt.ylim(0.45, 1.25)

plt.show()
