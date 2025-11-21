# 19. Write python Program to realize Fuzzy Sets arithmetic.
# 20. Write a Python code for designing fuzzy controller for FAN. Inputs: Temperature (T) and
# Humidity (H). Output: Fan Speed (S). Use triangular membership functions. Implement
# Takagi-Sugeno model rules. Calculate fan speed for given inputs.
# 21. In an automatic air conditioner controller, design fuzzy logic controller with 3 fuzzy sets for
# Temperature, Rotor, and Fan speed. Compute composition of relations R1∘R2 and crisp value
# of fan speed.

import numpy as np
import pandas as pd

# TRIANGULAR MEMBERSHIP FUNCTION
# A triangular membership function is defined by 3 points (a, b, c)
# It increases from a→b, then decreases from b→c.

def triangular_membership(x, a, b, c):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)

    # Left slope (increasing)
    increasing = (x >= a) & (x <= b)
    mu[increasing] = (x[increasing] - a) / (b - a + 1e-12)

    # Right slope (decreasing)
    decreasing = (x >= b) & (x <= c)
    mu[decreasing] = (c - x[decreasing]) / (c - b + 1e-12)

    # Peak (b)
    mu[x == b] = 1.0

    return np.clip(mu, 0, 1)


# 19. FUZZY SET ARITHMETIC

def fuzzy_complement(mu):
    return 1 - mu

def fuzzy_union(muA, muB):
    return np.maximum(muA, muB)

def fuzzy_intersection(muA, muB):
    return np.minimum(muA, muB)

def fuzzy_algebraic_sum(muA, muB):
    return muA + muB - muA * muB

def fuzzy_algebraic_product(muA, muB):
    return muA * muB

# Extension principle: fuzzy addition
def fuzzy_extension_add(universeA, muA, universeB, muB):
    result = {}
    for a, mu_a in zip(universeA, muA):
        for b, mu_b in zip(universeB, muB):
            z = round(a + b, 6)
            membership = min(mu_a, mu_b)
            result[z] = max(result.get(z, 0), membership)
    values = np.array(sorted(result.keys()))
    memberships = np.array([result[v] for v in values])
    return values, memberships

# Extension principle: fuzzy multiplication
def fuzzy_extension_multiply(universeA, muA, universeB, muB):
    result = {}
    for a, mu_a in zip(universeA, muA):
        for b, mu_b in zip(universeB, muB):
            z = round(a * b, 6)
            membership = min(mu_a, mu_b)
            result[z] = max(result.get(z, 0), membership)
    values = np.array(sorted(result.keys()))
    memberships = np.array([result[v] for v in values])
    return values, memberships


# 20. TAKAGI–SUGENO FAN CONTROLLER
# Inputs: Temperature (°C) and Humidity (%)
# Output: Fan Speed (RPM)
# Using triangular membership functions and TS rules.

# Temperature fuzzy sets
temperature_mfs = {
    "Low": lambda x: triangular_membership(x, 0, 0, 20),
    "Medium": lambda x: triangular_membership(x, 10, 20, 30),
    "High": lambda x: triangular_membership(x, 20, 40, 40)
}

# Humidity fuzzy sets
humidity_mfs = {
    "Low": lambda x: triangular_membership(x, 0, 0, 50),
    "Medium": lambda x: triangular_membership(x, 25, 50, 75),
    "High": lambda x: triangular_membership(x, 50, 100, 100)
}

# Takagi-Sugeno rule base
# Each rule outputs a FUNCTION of (T, H) → z
ts_rules = [
    ("Low", "Low", lambda T,H: 200),
    ("Low", "Medium", lambda T,H: 300),
    ("Low", "High", lambda T,H: 400),

    ("Medium", "Low", lambda T,H: 800 + 10*T - 2*H),
    ("Medium", "Medium", lambda T,H: 1500 + 8*T + 4*H),
    ("Medium", "High", lambda T,H: 1800 + 12*T + 6*H),

    ("High", "Low", lambda T,H: 2000 + 15*T + 1*H),
    ("High", "Medium", lambda T,H: 2400 + 10*T + 5*H),
    ("High", "High", lambda T,H: 2800 + 18*T + 10*H)
]

def compute_ts_fan_speed(temperature, humidity):
    rule_strengths = []
    rule_outputs = []

    for temp_label, hum_label, rule_function in ts_rules:
        mu_temp = temperature_mfs[temp_label](np.array([temperature]))[0]
        mu_hum = humidity_mfs[hum_label](np.array([humidity]))[0]

        firing_strength = min(mu_temp, mu_hum)
        rule_output = rule_function(temperature, humidity)

        rule_strengths.append(firing_strength)
        rule_outputs.append(rule_output)

    rule_strengths = np.array(rule_strengths)
    rule_outputs = np.array(rule_outputs)

    if rule_strengths.sum() == 0:
        return 0  # no activation

    # Weighted average (TS defuzzification)
    return float(np.sum(rule_strengths * rule_outputs) / rule_strengths.sum())


# 21. FUZZY RELATION COMPOSITION & CRISP FAN SPEED

# Define discrete temperature, rotor speed, and fan speed
temperature_values = np.array([18, 22, 26])
rotor_values = np.array([500, 1000, 1500])
fan_values = np.array([800, 1600, 2400])

# Fuzzy relation: Temperature → Rotor
R1 = np.array([
    [0.8, 0.4, 0.0],
    [0.2, 0.9, 0.3],
    [0.0, 0.5, 0.9]
])

# Fuzzy relation: Rotor → Fan
R2 = np.array([
    [0.9, 0.2, 0.0],
    [0.3, 0.8, 0.4],
    [0.0, 0.4, 0.95]
])

# Max-min composition of fuzzy relations
def max_min_composition(R1, R2):
    rows, shared_dim = R1.shape
    _, cols = R2.shape
    result = np.zeros((rows, cols))

    for i in range(rows):
        for k in range(cols):
            result[i, k] = max(min(R1[i, j], R2[j, k]) for j in range(shared_dim))

    return result

R_composed = max_min_composition(R1, R2)

# Fuzzify crisp temperature for relation-based inference
def fuzzify_temperature_crisp(t, centers):
    memberships = []
    for i, center in enumerate(centers):
        if i == 0:
            a,b,c2 = centers[0]-4, centers[0], centers[1]
        elif i == len(centers)-1:
            a,b,c2 = centers[-2], centers[-1], centers[-1]+4
        else:
            a,b,c2 = centers[i-1], centers[i], centers[i+1]

        mu = triangular_membership(np.array([t]), a, b, c2)[0]
        memberships.append(mu)

    memberships = np.array(memberships)

    if memberships.sum() != 0:
        memberships /= memberships.sum()

    return memberships

# Infer fan fuzzy set
def infer_fan_from_temp(mu_temp, R):
    return np.array([
        max(min(mu_temp[i], R[i, k]) for i in range(len(mu_temp)))
        for k in range(R.shape[1])
    ])

# Defuzzification using centroid method
def centroid(values, memberships):
    if memberships.sum() == 0:
        return float(values.mean())
    return float(np.sum(values * memberships) / memberships.sum())


# Example Execution

if __name__ == "__main__":

    # TS controller example
    print("Takagi-Sugeno Fan Speed:", compute_ts_fan_speed(25, 40))

    # Relation composition
    print("\nR1 ∘ R2:")
    print(R_composed)

    # Crisp temperature example
    mu_temp = fuzzify_temperature_crisp(23, temperature_values)
    mu_fan = infer_fan_from_temp(mu_temp, R_composed)
    crisp_fan_speed = centroid(fan_values, mu_fan)

    print("\nFuzzified Temperature:", mu_temp)
    print("Fan Fuzzy Set:", mu_fan)
    print("Crisp Fan Speed:", crisp_fan_speed)
