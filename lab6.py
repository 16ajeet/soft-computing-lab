# 19. Write python Program to realize Fuzzy Sets arithmetic.
# 20. Write a Python code for designing fuzzy controller for FAN. Inputs: Temperature (T) and
# Humidity (H). Output: Fan Speed (S). Use triangular membership functions. Implement
# Takagi-Sugeno model rules. Calculate fan speed for given inputs.
# 21. In an automatic air conditioner controller, design fuzzy logic controller with 3 fuzzy sets for
# Temperature, Rotor, and Fan speed. Compute composition of relations R1∘R2 and crisp value
# of fan speed.

def tri(x, a, b, c):  # Triangular membership
    return max(0, min((x-a)/(b-a) if b>a else 0, (c-x)/(c-b) if c>b else 0))

def fuzzy_set(universe, params):  # params = [a,b,c] for triangle
    return [tri(x, *params) for x in universe]

def fuzzy_union(A, B): return [max(a,b) for a,b in zip(A,B)]
def fuzzy_intersect(A, B): return [min(a,b) for a,b in zip(A,B)]
def fuzzy_complement(A): return [1-a for a in A]

universe = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

# Define two fuzzy sets A and B
A = fuzzy_set(universe, [1,2,3])  # A: triangle(1,2,3)
B = fuzzy_set(universe, [2,3,4])  # B: triangle(2,3,4)

print("Universe:", universe)
print("A:", [f"{a:.2f}" for a in A])
print("B:", [f"{b:.2f}" for b in B])
print("A ∪ B:", [f"{x:.2f}" for x in fuzzy_union(A,B)])
print("A ∩ B:", [f"{x:.2f}" for x in fuzzy_intersect(A,B)])
print("A'", [f"{x:.2f}" for x in fuzzy_complement(A)])

#==================================================
def tri(x, a, b, c):
    return max(0, min((x - a) / (b - a) if b > a else 0, (c - x) / (c - b) if c > b else 0))

def fuzzify_temp(temp):
    return {
        'Low': tri(temp, 0, 0, 25),
        'Medium': tri(temp, 15, 30, 45),
        'High': tri(temp, 35, 50, 50)
    }

def fuzzify_hum(hum):
    return {
        'Low': tri(hum, 0, 0, 40),
        'Medium': tri(hum, 30, 50, 70),
        'High': tri(hum, 60, 80, 80)
    }

# Takagi-Sugeno fuzzy rules: output is a linear function or constant
rules = [
    (('Low', 'Low'), 10),
    (('Low', 'Medium'), 20),
    (('Low', 'High'), 30),
    (('Medium', 'Low'), 40),
    (('Medium', 'Medium'), 60),
    (('Medium', 'High'), 80),
    (('High', 'Low'), 70),
    (('High', 'Medium'), 90),
    (('High', 'High'), 100)
]

def fuzzy_controller(temp, hum):
    temp_mf = fuzzify_temp(temp)
    hum_mf = fuzzify_hum(hum)
    numerator = 0
    denominator = 0
    for (t_label, h_label), output in rules:
        weight = temp_mf[t_label] * hum_mf[h_label]
        numerator += weight * output
        denominator += weight
    return numerator / denominator if denominator != 0 else 0

# Example inputs
T = 28  # Temperature
H = 45  # Humidity

speed = fuzzy_controller(T, H)
print(f"Fan speed for Temperature={T} and Humidity={H} is {speed:.2f}")


#===========================================================
def tri(x, a, b, c):
    return max(0, min((x - a) / (b - a) if b > a else 0, (c - x) / (c - b) if c > b else 0))

# Define fuzzy sets (example ranges)
temp_sets = {
    'Low': (0, 0, 25),
    'Medium': (15, 30, 45),
    'High': (35, 50, 50)
}

rotor_sets = {
    'Slow': (0, 0, 5),
    'Medium': (3, 5, 7),
    'Fast': (6, 10, 10)
}

fan_speed_sets = {
    'Low': 10,
    'Medium': 50,
    'High': 90
}

# Membership for input Temperature
def temp_membership(temp):
    return {key: tri(temp, *val) for key, val in temp_sets.items()}

# Membership for Rotor speed
def rotor_membership(rotor):
    return {key: tri(rotor, *val) for key, val in rotor_sets.items()}

# Define relations R1: Temp->Rotor (membership degrees)
R1 = {
    ('Low', 'Slow'): 0.8,
    ('Low', 'Medium'): 0.4,
    ('Medium', 'Medium'): 0.7,
    ('Medium', 'Fast'): 0.3,
    ('High', 'Fast'): 0.9,
    ('High', 'Medium'): 0.5
}

# Define relations R2: Rotor->Fan speed (membership degrees)
R2 = {
    ('Slow', 'Low'): 0.7,
    ('Medium', 'Medium'): 0.8,
    ('Fast', 'High'): 0.9
}

# Composition R1 ∘ R2 (max-min composition)
def compose_relation(R1, R2):
    composite = {}
    for (t, r1_val) in R1.items():
        for (r2_key, f) in R2.items():
            if t[1] == r2_key[0]:
                key = (t[0], r2_key[1])
                val = min(R1[t], R2[r2_key])
                composite[key] = max(composite.get(key, 0), val)
    return composite

composite = compose_relation(R1, R2)

# Defuzzify fan speed using weighted average
fan_speed_vals = fan_speed_sets
numerator = 0
denominator = 0
for (temp_label, fan_label), degree in composite.items():
    speed = fan_speed_vals[fan_label]
    numerator += degree * speed
    denominator += degree

crisp_fan_speed = numerator / denominator if denominator != 0 else 0

print(f"Composite relation degrees: {composite}")
print(f"Crisp fan speed value: {crisp_fan_speed:.2f}")
