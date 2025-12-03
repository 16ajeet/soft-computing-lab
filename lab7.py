# 22. Write a Python code to implement various defuzzification methods (menu-driven): Centroid,
# Center of Sums, Weighted Sum, Mean of Maximum, Smallest of Maximum, Largest of
# Maximum. Use trapezoidal functions (overlapping, user-defined).
# 23. Design a fuzzy logic controller for a washing machine using Python. Inputs: Degree of dirt,
# Type of dirt. Output: Wash time. Implement rules for wash time (Very Short, Short, Medium,
# Long, Very Long).

# ===========================
# Part 22: Defuzzification
# ===========================

def trapezoid(x, a, b, c, d):
    """Trapezoidal membership function with safe division."""
    if x < a or x > d:
        return 0.0
    if a <= x < b:
        return (x - a) / (b - a) if b != a else 1.0  # handle a == b
    if b <= x <= c:
        return 1.0
    if c < x <= d:
        return (d - x) / (d - c) if d != c else 1.0  # handle c == d
    return 0.0


def defuzzify(method, points, values):
    """Various defuzzification methods for a single aggregated fuzzy set."""
    if len(points) != len(values):
        raise ValueError("points and values must have same length")

    if method == "centroid":
        numerator = sum(x * v for x, v in zip(points, values))
        denominator = sum(values)
        return numerator / denominator if denominator != 0 else 0.0

    elif method == "center_of_sums":
        # crude left/right split as you wrote
        mid = len(values) // 2
        left_vals = values[:mid]
        right_vals = values[mid:]
        left_pts = points[:mid]
        right_pts = points[mid:]

        left_area = sum(left_vals)
        right_area = sum(right_vals)

        left_centroid = (sum(left_pts[i] * left_vals[i] for i in range(len(left_vals))) / left_area) if left_area else 0.0
        right_centroid = (sum(right_pts[i] * right_vals[i] for i in range(len(right_vals))) / right_area) if right_area else 0.0

        if left_area == 0 and right_area == 0:
            return 0.0
        elif left_area == 0:
            return right_centroid
        elif right_area == 0:
            return left_centroid
        else:
            return (left_centroid + right_centroid) / 2.0

    elif method == "weighted_sum":
        # note: no normalization, this is not centroid
        return sum(points[i] * values[i] for i in range(len(points)))

    elif method == "mean_of_maximum":
        max_val = max(values)
        if max_val == 0:
            return 0.0
        max_points = [points[i] for i, v in enumerate(values) if v == max_val]
        return sum(max_points) / len(max_points)

    elif method == "smallest_of_maximum":
        max_val = max(values)
        if max_val == 0:
            return 0.0
        return min(points[i] for i, v in enumerate(values) if v == max_val)

    elif method == "largest_of_maximum":
        max_val = max(values)
        if max_val == 0:
            return 0.0
        return max(points[i] for i, v in enumerate(values) if v == max_val)

    else:
        raise ValueError("Unknown method: " + method)


# Example usage with trapezoidal fuzzy set
points = [0, 1, 2, 3, 4, 5]
values = [trapezoid(x, 1, 2, 3, 4) for x in points]

method = input("Enter defuzzification method (centroid, center_of_sums, weighted_sum, mean_of_maximum, smallest_of_maximum, largest_of_maximum): ").strip()
result = defuzzify(method, points, values)
print(f"Defuzzified value ({method}): {result}")


# ===========================
# Part 23: Washing machine FLC
# ===========================

def tri(x, a, b, c):
    """
    Triangular membership function supporting shoulder triangles:
    a <= b <= c, can have a == b or b == c.
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    elif b <= x < c:
        return (c - x) / (c - b) if c != b else 1.0
    else:
        return 0.0


def fuzzify_dirt(degree):  # Low, Medium, High
    # shoulder on left (0,0,3), center (0,3,6), right shoulder (3,6,10)
    return [
        tri(degree, 0, 0, 3),   # Low
        tri(degree, 0, 3, 6),   # Medium
        tri(degree, 3, 6, 10)   # High
    ]


def fuzzify_type(typ):  # Easy, Normal, Hard
    return [
        tri(typ, 0, 0, 3),      # Easy
        tri(typ, 0, 3, 6),      # Normal
        tri(typ, 3, 6, 10)      # Hard
    ]


def rules(dirt, typ):
    """
    dirt = [low, med, high]
    typ  = [easy, normal, hard]
    wash_time = [VShort, Short, Med, Long, VLong]
    """
    wash_time = [0.0] * 5  # VShort, Short, Med, Long, VLong

    # Very Short: low dirt + easy type
    wash_time[0] = min(dirt[0], typ[0])

    # Short: (low & normal) OR (medium & easy)
    wash_time[1] = max(
        wash_time[1],
        min(dirt[0], typ[1]),
        min(dirt[1], typ[0])
    )

    # Medium: (medium & normal) OR (high & easy)
    wash_time[2] = max(
        wash_time[2],
        min(dirt[1], typ[1]),
        min(dirt[2], typ[0])
    )

    # Long: (high & normal) OR (medium & hard)
    wash_time[3] = max(
        wash_time[3],
        min(dirt[2], typ[1]),
        min(dirt[1], typ[2])
    )

    # Very Long: high dirt + hard type
    wash_time[4] = min(dirt[2], typ[2])

    return wash_time


def defuzzify_washing(wt):
    """
    wt: [μ_VShort, μ_Short, μ_Med, μ_Long, μ_VLong]
    Defuzzify with weighted average.
    """
    times = [1, 3, 5, 8, 12]  # Minutes for each level
    s = sum(wt)
    return (sum(t * w for t, w in zip(times, wt)) / s) if s != 0 else 5.0


# Test
degree = 7  # Degree of dirt (0-10)
typ = 4     # Type of dirt (0-10)

dirt_mf = fuzzify_dirt(degree)
typ_mf = fuzzify_type(typ)
wash_time = rules(dirt_mf, typ_mf)
result = defuzzify_washing(wash_time)

print(f"Dirt: {degree}, Type: {typ}")
print("Dirt MF [Low, Med, High]:", dirt_mf)
print("Type MF [Easy, Normal, Hard]:", typ_mf)
print("Wash time MF [VShort, Short, Med, Long, VLong]:", wash_time)
print(f"Wash time: {result:.1f} minutes")
