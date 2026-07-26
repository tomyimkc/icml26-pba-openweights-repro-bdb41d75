import numpy as np
import json

# Set fixed seed for reproducibility (required by contract)
rng = np.random.default_rng(0)
T = 1000  # Number of steps (largest feasible scale under 5 minutes)
t = 0.5   # True root position

# Symmetric noise model (paper's assumed condition)
L_sym, R_sym = 0.0, 1.0
errors_sym = []
signs_sym = []

for _ in range(T):
    x = (L_sym + R_sym) / 2.0
    errors_sym.append(abs(x - t))
    signs_sym.append(1 if x > t else -1)
    
    true_response = 1 if x > t else -1
    if true_response == 1:
        Y = 1 if rng.random() < 0.9 else 0
    else:
        Y = 1 if rng.random() < 0.1 else 0  # Symmetric: P(Y=1|negative)=0.1
    
    if Y == 1:
        R_sym = x
    else:
        L_sym = x

# Compute symmetric metrics (skip first ratio due to zero division)
ratios_sym = [errors_sym[i+1] / errors_sym[i] for i in range(1, T-1)]
avg_ratio_sym = float(np.mean(ratios_sym)) if len(ratios_sym) > 0 else 0.0
sign_changes_sym = sum(1 for i in range(1, T) if signs_sym[i] != signs_sym[i-1])

# Asymmetric noise control (violates symmetric noise precondition)
L_asym, R_asym = 0.0, 1.0
errors_asym = []
signs_asym = []

for _ in range(T):
    x = (L_asym + R_asym) / 2.0
    errors_asym.append(abs(x - t))
    signs_asym.append(1 if x > t else -1)
    
    true_response = 1 if x > t else -1
    if true_response == 1:
        Y = 1 if rng.random() < 0.9 else 0
    else:
        Y = 1 if rng.random() < 0.9 else 0  # Asymmetric: P(Y=1|negative)=0.9
    
    if Y == 1:
        R_asym = x
    else:
        L_asym = x

# Compute asymmetric metrics (skip first ratio due to zero division)
ratios_asym = [errors_asym[i+1] / errors_asym[i] for i in range(1, T-1)]
avg_ratio_asym = float(np.mean(ratios_asym)) if len(ratios_asym) > 0 else 0.0
sign_changes_asym = sum(1 for i in range(1, T) if signs_asym[i] != signs_asym[i-1])

# Output results in required JSON format
print("RESULT_JSON " + json.dumps({
    "avg_ratio_sym": avg_ratio_sym,
    "sign_changes_sym": sign_changes_sym,
    "avg_ratio_asym": avg_ratio_asym,
    "sign_changes_asym": sign_changes_asym
}))
