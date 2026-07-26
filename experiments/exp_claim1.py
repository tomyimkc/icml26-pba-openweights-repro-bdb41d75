import numpy as np
import json

# Seed everything for determinism (critical for reproducibility)
np.random.seed(0)  # Base seed for reproducibility

# Problem setup (root at 0.37 to avoid alignment with initial median 0.5)
x0 = 0.37
T = 1000  # Maximum steps (maximizing resolution within time budget)

# Define p values to test (main case, relaxed precondition, noiseless control)
p_values = [0.9, 0.51, 1.0]
p_key_map = {0.9: "p09", 0.51: "p051", 1.0: "p10"}

# Track ratios across seeds for each p
ratios_by_p = {p: [] for p in p_values}

# Run for 10 seeds (0-9) to ensure results are outside noise variance
for seed in range(10):
    rng = np.random.default_rng(seed)
    for p in p_values:
        # Initialize belief: uniform distribution over [0,1]
        breakpoints = [0.0, 1.0]  # Sorted boundaries
        weights = [1.0]  # Unnormalized density per interval
        errors = []  # Store error after each step (before update)
        
        for _ in range(T):
            # Compute current median (query point)
            total_mass = 0.0
            for i in range(len(breakpoints) - 1):
                width = breakpoints[i+1] - breakpoints[i]
                total_mass += weights[i] * width
            
            # Find median where cumulative mass reaches total_mass/2
            cum_mass = 0.0
            x_t = breakpoints[-1]  # Fallback
            for i in range(len(breakpoints) - 1):
                width = breakpoints[i+1] - breakpoints[i]
                mass_i = weights[i] * width
                if mass_i < 1e-15:  # Skip zero-mass intervals
                    continue
                if cum_mass + mass_i >= total_mass / 2.0:
                    fraction = (total_mass / 2.0 - cum_mass) / mass_i
                    x_t = breakpoints[i] + fraction * width
                    break
                cum_mass += mass_i
            
            # Record error (distance from median to root)
            error = abs(x_t - x0)
            errors.append(error)
            
            # Generate noisy oracle response
            true_direction = "right" if x_t < x0 else "left"
            if rng.random() < p:
                response = true_direction
            else:
                response = "left" if true_direction == "right" else "right"
            
            # Insert x_t as breakpoint if not present (with tolerance)
            tol = 1e-15
            if not any(abs(x_t - bp) < tol for bp in breakpoints):
                idx = 0
                while idx < len(breakpoints) and breakpoints[idx] < x_t:
                    idx += 1
                breakpoints.insert(idx, x_t)
                weights.insert(idx, weights[idx-1])
            
            # Update weights based on response
            if response == "right":
                factor_left = 1 - p
                factor_right = p
            else:  # "left"
                factor_left = p
                factor_right = 1 - p
            
            for i in range(len(breakpoints) - 1):
                if breakpoints[i+1] <= x_t + tol:  # Left piece
                    weights[i] *= factor_left
                elif breakpoints[i] >= x_t - tol:  # Right piece
                    weights[i] *= factor_right
            
            # Renormalize weights to total mass 1.0
            total_mass = 0.0
            for i in range(len(breakpoints) - 1):
                width = breakpoints[i+1] - breakpoints[i]
                total_mass += weights[i] * width
            if total_mass < 1e-15:
                total_mass = 1.0
            for i in range(len(weights)):
                weights[i] /= total_mass
        
        # Compute log-linear decay slope (geometric ratio) for this seed
        t_list = []
        log_errors = []
        for t in range(10, T-10):  # Skip initial transient and noise floor
            if errors[t] > 1e-15:  # Avoid log(0)
                t_list.append(t)
                log_errors.append(np.log(errors[t]))
        
        if len(t_list) >= 2:
            t_arr = np.array(t_list)
            log_err_arr = np.array(log_errors)
            # Fit log(error) = a*t + b -> ratio = exp(a)
            a, _ = np.polyfit(t_arr, log_err_arr, 1)
            ratio = np.exp(a)
            ratios_by_p[p].append(ratio)

# Compute mean and std for each p across seeds
result = {}
for p in p_values:
    ratios = ratios_by_p[p]
    if ratios:
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
    else:
        mean_ratio = float('nan')
        std_ratio = float('nan')
    
    p_key = p_key_map[p]
    result[f"pba_{p_key}_ratio_mean"] = mean_ratio
    result[f"pba_{p_key}_ratio_std"] = std_ratio

# Output in required format
print("RESULT_JSON " + json.dumps(result))
