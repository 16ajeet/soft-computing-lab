# 19. Write python Program to realize Fuzzy Sets arithmetic.
# 20. Write a Python code for designing fuzzy controller for FAN. Inputs: Temperature (T) and
# Humidity (H). Output: Fan Speed (S). Use triangular membership functions. Implement
# Takagi-Sugeno model rules. Calculate fan speed for given inputs.
# 21. In an automatic air conditioner controller, design fuzzy logic controller with 3 fuzzy sets for
# Temperature, Rotor, and Fan speed. Compute composition of relations R1∘R2 and crisp value
# of fan speed.

import numpy as np
import pandas as pd

# ===========================================================
# TRIANGULAR MEMBERSHIP FUNCTION
# ===========================================================

def triangular_membership(x, a, b, c):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)

    inc = (x >= a) & (x <= b)
    mu[inc] = (x[inc] - a) / (b - a + 1e-12)

    dec = (x >= b) & (x <= c)
    mu[dec] = (c - x[dec]) / (c - b + 1e-12)

    mu[x == b] = 1.0
    return np.clip(mu, 0, 1)


# ===========================================================
# 19. FUZZY SET OPERATIONS
# ===========================================================

def fuzzy_complement(mu): return 1 - mu
def fuzzy_union(muA, muB): return np.maximum(muA, muB)
def fuzzy_intersection(muA, muB): return np.minimum(muA, muB)
def fuzzy_algebraic_sum(muA, muB): return muA + muB - muA * muB
def fuzzy_algebraic_product(muA, muB): return muA * muB

def fuzzy_extension_add(valsA, muA, valsB, muB):
    result = {}
    for a, ma in zip(valsA, muA):
        for b, mb in zip(valsB, muB):
            z = round(a + b, 6)
            result[z] = max(result.get(z, 0), min(ma, mb))
    return np.array(sorted(result)), np.array([result[z] for z in sorted(result)])

def fuzzy_extension_multiply(valsA, muA, valsB, muB):
    result = {}
    for a, ma in zip(valsA, muA):
        for b, mb in zip(valsB, muB):
            z = round(a * b, 6)
            result[z] = max(result.get(z, 0), min(ma, mb))
    return np.array(sorted(result)), np.array([result[z] for z in sorted(result)])

# ===========================================================
# 20. TAKAGI–SUGENO FAN CONTROLLER
# ===========================================================

temperature_mfs = {
    "Low": lambda x: triangular_membership(x, 0, 0, 20),
    "Medium": lambda x: triangular_membership(x, 10, 20, 30),
    "High": lambda x: triangular_membership(x, 20, 40, 40)
}

humidity_mfs = {
    "Low": lambda x: triangular_membership(x, 0, 0, 50),
    "Medium": lambda x: triangular_membership(x, 25, 50, 75),
    "High": lambda x: triangular_membership(x, 50, 100, 100)
}

ts_rules = [
    ("Low", "Low", lambda T,H: 200),
    ("Low", "Medium", lambda T,H: 300),
    ("Low", "High", lambda T,H: 400),
    ("Medium", "Low", lambda T,H: 800 + 10*T - 2*H),
    ("Medium", "Medium", lambda T,H: 1500 + 8*T + 4*H),
    ("Medium", "High", lambda T,H: 1800 + 12*T + 6*H),
    ("High", "Low", lambda T,H: 2000 + 15*T + 1*H),
    ("High", "Medium", lambda T,H: 2400 + 10*T + 5*H),
    ("High", "High", lambda T,H: 2800 + 18*T + 10*H),
]

def compute_ts_fan_speed(T, H):
    strengths = []
    outputs = []

    for temp_label, hum_label, fn in ts_rules:
        mu_T = temperature_mfs[temp_label](np.array([T]))[0]
        mu_H = humidity_mfs[hum_label](np.array([H]))[0]

        firing = min(mu_T, mu_H)
        strengths.append(firing)
        outputs.append(fn(T, H))

    strengths = np.array(strengths)
    outputs = np.array(outputs)

    if strengths.sum() == 0:
        return 0

    return float(np.sum(strengths * outputs) / strengths.sum())


# ===========================================================
# 21. FUZZY RELATION COMPOSITION
# ===========================================================

Temp_vals = np.array([18, 22, 26])
Rotor_vals = np.array([500, 1000, 1500])
Fan_vals = np.array([800, 1600, 2400])

R1 = np.array([
    [0.8, 0.4, 0.0],
    [0.2, 0.9, 0.3],
    [0.0, 0.5, 0.9]
])

R2 = np.array([
    [0.9, 0.2, 0.0],
    [0.3, 0.8, 0.4],
    [0.0, 0.4, 0.95]
])

def max_min_composition(R1, R2):
    out = np.zeros((R1.shape[0], R2.shape[1]))
    for i in range(R1.shape[0]):
        for k in range(R2.shape[1]):
            out[i, k] = max(min(R1[i, j], R2[j, k]) for j in range(R1.shape[1]))
    return out

R_comp = max_min_composition(R1, R2)

def fuzzify_temp_crisp(t):
    mus = []
    for i, c in enumerate(Temp_vals):
        if i == 0:
            a,b,c2 = Temp_vals[0]-4, Temp_vals[0], Temp_vals[1]
        elif i == 2:
            a,b,c2 = Temp_vals[1], Temp_vals[2], Temp_vals[2]+4
        else:
            a,b,c2 = Temp_vals[i-1], Temp_vals[i], Temp_vals[i+1]

        mu = triangular_membership(np.array([t]), a, b, c2)[0]
        mus.append(mu)

    mus = np.array(mus)
    if mus.sum() != 0:
        mus = mus / mus.sum()
    return mus

def infer_fan(mu_temp, R):
    return np.array([max(min(mu_temp[i], R[i, k]) for i in range(len(mu_temp))) 
                     for k in range(R.shape[1])])

def centroid(values, mus):
    if mus.sum() == 0:
        return values.mean()
    return np.sum(values * mus) / mus.sum()


# ===========================================================
# PRINT EVERYTHING
# ===========================================================

if __name__ == "__main__":

    print("\n==============================")
    print(" 19. FUZZY SET OPERATIONS")
    print("==============================")

    universe = np.array([0, 1, 2, 3, 4, 5])
    muA = triangular_membership(universe, 0, 2, 4)
    muB = triangular_membership(universe, 1, 3, 5)

    print("\nUniverse:", universe)
    print("muA:", muA)
    print("muB:", muB)
    print("\nComplement of A:", fuzzy_complement(muA))
    print("Union A U B:", fuzzy_union(muA, muB))
    print("Intersection A ∩ B:", fuzzy_intersection(muA, muB))
    print("Algebraic Sum:", fuzzy_algebraic_sum(muA, muB))
    print("Algebraic Product:", fuzzy_algebraic_product(muA, muB))

    add_vals, add_mu = fuzzy_extension_add(universe, muA, universe, muB)
    mul_vals, mul_mu = fuzzy_extension_multiply(universe, muA, universe, muB)

    print("\nExtension Principle Addition:")
    print("Values:", add_vals)
    print("Memberships:", add_mu)

    print("\nExtension Principle Multiplication:")
    print("Values:", mul_vals)
    print("Memberships:", mul_mu)


    print("\n==============================")
    print(" 20. TAKAGI–SUGENO FAN CONTROLLER")
    print("==============================")

    print("Fan speed for T=25, H=40 →", compute_ts_fan_speed(25, 40))


    print("\n==============================")
    print(" 21. RELATION COMPOSITION")
    print("==============================")

    print("\nR1:\n", R1)
    print("\nR2:\n", R2)
    print("\nR1 ∘ R2:\n", R_comp)

    mu_temp = fuzzify_temp_crisp(23)
    print("\nFuzzified temperature 23°C:", mu_temp)

    mu_fan = infer_fan(mu_temp, R_comp)
    print("Fan fuzzy set:", mu_fan)

    crisp_speed = centroid(Fan_vals, mu_fan)
    print("Crisp Fan Speed:", crisp_speed)
