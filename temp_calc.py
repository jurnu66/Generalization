import numpy as np
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

def expectation(func):
    res, _ = integrate.dblquad(lambda x2, x1: func(x1, x2) * 0.25, -1, 1, lambda x1: -1, lambda x1: 1)
    return res

def expected_var(var_func_x):
    res, _ = integrate.quad(lambda x: var_func_x(x) * 0.5, -1, 1)
    return res

def analyze(target_name, f):
    print(f"--- Analytical for {target_name} ---")
    E_fx2, _ = integrate.quad(lambda x: f(x)**2 * 0.5, -1, 1)

    # 1. Constant Model: h(x) = b
    # b = (f(x1) + f(x2))/2
    b_mean = expectation(lambda x1, x2: (f(x1) + f(x2))/2)
    b_var = expectation(lambda x1, x2: ((f(x1) + f(x2))/2 - b_mean)**2)
    
    bias2_const = expected_var(lambda x: (b_mean - f(x))**2)
    var_const = b_var
    print(f"Constant Model : Bias2 = {bias2_const:.4f}, Var = {var_const:.4f}, E_out = {bias2_const+var_const:.4f}")

    # 2. Linear Model: h(x) = ax + b
    # a = (f(x2) - f(x1))/(x2 - x1)
    # b = f(x1) - a*x1 = (x2*f(x1) - x1*f(x2))/(x2 - x1)
    a_mean = expectation(lambda x1, x2: (f(x2) - f(x1))/(x2 - x1) if x2 != x1 else 0)
    b_mean_lin = expectation(lambda x1, x2: (x2*f(x1) - x1*f(x2))/(x2 - x1) if x2 != x1 else 0)
    
    bias2_lin = expected_var(lambda x: (a_mean*x + b_mean_lin - f(x))**2)
    
    def var_lin_x(x):
        return expectation(lambda x1, x2: ( ((f(x2)-f(x1))/(x2-x1))*x + ((x2*f(x1)-x1*f(x2))/(x2-x1)) - (a_mean*x + b_mean_lin) )**2 if x2 != x1 else 0)
    
    var_lin = expected_var(var_lin_x)
    print(f"Linear Model   : Bias2 = {bias2_lin:.4f}, Var = {var_lin:.4f}, E_out = {bias2_lin+var_lin:.4f}")

    # 3. Linear Through Origin: h(x) = ax
    # a = (x1*f(x1) + x2*f(x2))/(x1**2 + x2**2)
    a_orig_mean = expectation(lambda x1, x2: (x1*f(x1) + x2*f(x2))/(x1**2 + x2**2) if (x1**2+x2**2) != 0 else 0)
    bias2_orig = expected_var(lambda x: (a_orig_mean*x - f(x))**2)
    
    def var_orig_x(x):
        return expectation(lambda x1, x2: ( ((x1*f(x1)+x2*f(x2))/(x1**2+x2**2))*x - a_orig_mean*x )**2 if (x1**2+x2**2) != 0 else 0)
        
    var_orig = expected_var(var_orig_x)
    print(f"Linear Origin  : Bias2 = {bias2_orig:.4f}, Var = {var_orig:.4f}, E_out = {bias2_orig+var_orig:.4f}")
    print()

analyze("sin(pi*x)", lambda x: np.sin(np.pi*x))
analyze("x^2", lambda x: x**2)

