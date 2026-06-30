import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from time import time

# Input --------------------------------------------------------------------------------------------
iron_core = True
tracker_res = 100e-6
length = 3.0
field = 1.0
momentum = 100

# compute scattering parameter
radiation_length = 17.6e-3 if iron_core else 300
s = 1/3 * (15e-3/momentum * length/tracker_res)**2 * (length/radiation_length)


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

# Define initial point -----------------------------------------------------------------------------
c = np.array([+1, 0, -1])
x = np.array([.5, 0, .5])
x0 = np.concatenate([c,x])
np.random.seed(0)
x0 += np.random.randn(6)*1e-3
n = x0.size // 2

# Define constraints -------------------------------------------------------------------------------
# nonlinear constraint
constraintN = lambda x: x[:n] @ x[n:]
nconst = NonlinearConstraint(constraintN, 0, 0)
# linear constraint
constraintL = np.concatenate([np.zeros(n), np.ones(n)])
lconst = LinearConstraint(constraintL, 1, 1)
# max and min bound for variables
boundmin = np.concatenate([-np.inf*np.ones(n), np.zeros(n)])
boundmax = np.concatenate([+np.inf*np.ones(n), np. ones(n)])
bounds = Bounds(boundmin, boundmax)

# Optimization -------------------------------------------------------------------------------------
obj_fun = lambda x: resolution(x[:n], x[n:], s)
opt = {"ftol": 1e-12, "eps": 1e-8, "maxiter": 1000}
tstart = time()
sol = minimize(obj_fun, x0, method="slsqp", options=opt, bounds=bounds, constraints=[lconst,nconst])
tend = time()
c = sol.x[:n]; x = sol.x[n:]; norm_res = sol.fun

# Print optimization results -----------------------------------------------------------------------
print(f"\nOptimization finished in {tend - tstart:.2f} seconds")
print(f"{sol.message}\n")
print("Normalized Results:")
print(f"Scattering parameter: {s:.3e}")
print(f"Trackers spacing: {x[0]:.3f}  {x[1]:.3f}  {x[2]:.3f}")
print(f"Coefficients: {c[0]:.3f}  {c[1]:.3f}  {c[2]:.3f}")
print(f"Normalized resolution: {norm_res:.3f}\n")

x = x*length
abs_res = norm_res * np.sqrt(s+16) * (10/3 * 2 * momentum * tracker_res / (field * length**2))
print("Absolute Results:")
print(f"Trackers spacing: {x[0]:.3f}  {x[1]:.3f}  {x[2]:.3f}")
print(f"Coefficients: {c[0]:.3f}  {c[1]:.3f}  {c[2]:.3f}")
print(f"Resolution: {abs_res:.3f}")
