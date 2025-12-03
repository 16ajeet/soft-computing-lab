import numpy as np

def euclid_dist(x, w):
    return np.sum((x - w)**2)

# ------------------ Kohonen SOM (simple example) ------------------
np.random.seed(0)
inputs = np.random.rand(10, 5)  # Example input dataset (10 samples, 5 features)
weights = np.random.rand(2, 5)  # 2 output neurons with weights

lr = 0.1
epochs = 10

for _ in range(epochs):
    for x in inputs:
        # Find winning neuron (min distance)
        distances = [euclid_dist(x, w) for w in weights]
        winner = np.argmin(distances)
        # Update winner weights
        weights[winner] += lr * (x - weights[winner])

print("Final weights of Kohonen layer:\n", weights)

# ------------------ Counterpropagation Network (CPN) ------------------
# Network: 4 inputs -> 2 Kohonen -> 2 Grossberg
X = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
Y = np.array([[1,0],[0,1],[1,0],[0,1]])

# Initial weights
W_kohonen = np.random.rand(2,4)*0.1  # 2x4
W_grossberg = np.random.rand(2,2)*0.1  # 2x2

lr1 = lr2 = 0.5

print("Initial Kohonen weights:\n", W_kohonen)
print("Initial Grossberg weights:\n", W_grossberg)

# Phase I: Unsupervised (Kohonen)
for x in X:
    dists = [euclid_dist(x, w) for w in W_kohonen]
    winner = np.argmin(dists)
    W_kohonen[winner] += lr1 * (x - W_kohonen[winner])
    print(f"Input {x}, Winner: {winner}")

print("\nKohonen weights after Phase I:\n", W_kohonen)

# Phase II: Supervised (Grossberg)
for x, y in zip(X, Y):
    dists = [euclid_dist(x, w) for w in W_kohonen]
    winner = np.argmin(dists)
    h = np.zeros(2); h[winner] = 1
    e = y - np.dot(W_grossberg, h)
    W_grossberg += lr2 * np.outer(e, h)

print("Final Grossberg weights:\n", W_grossberg)

# Test
print("\nFinal outputs:")
for x, y in zip(X, Y):
    dists = [euclid_dist(x, w) for w in W_kohonen]
    winner = np.argmin(dists)
    h = np.zeros(2); h[winner] = 1
    out = np.dot(W_grossberg, h)
    print(f"Input: {x}, Target: {y}, Output: {out}")

# ------------------ ART1 and ART2 (fixed to avoid infinite loops) ------------------

def art1_train(X, vigilance=0.8, n_cat=5):
    """ART1: Binary ART with limited tries per pattern.

    If no category meets vigilance after trying all categories, the
    best-matching category is updated (to avoid infinite loops).
    """
    n_feat = len(X[0])
    W = np.random.rand(n_feat, n_cat)

    for x in X:
        x_norm = x / (np.sum(x) + 1e-12)
        tried = set()
        while True:
            scores = [np.sum(np.minimum(W[:, j], x_norm)) for j in range(n_cat)]
            cand_scores = [scores[j] if j not in tried else -1 for j in range(n_cat)]
            J = int(np.argmax(cand_scores))

            if J in tried or cand_scores[J] < 0:
                # no untried candidates left -> accept best overall
                J_best = int(np.argmax(scores))
                W[:, J_best] = np.maximum(W[:, J_best], x_norm)
                break

            rho = scores[J] / (np.sum(x_norm) + 1e-12)
            if rho >= vigilance:
                W[:, J] = np.maximum(W[:, J], x_norm)
                break
            else:
                tried.add(J)
                W[:, J] = np.zeros(n_feat)

    return W


def art2_train(X, vigilance=0.7, n_cat=5):
    """ART2: Continuous ART with limited tries per pattern."""
    n_feat = len(X[0])
    W = np.random.rand(n_feat*2, n_cat)*0.1

    for x in X:
        I = np.append(x, np.zeros(n_feat))
        tried = set()
        while True:
            scores = [np.dot(I, W[:, j]) for j in range(n_cat)]
            cand_scores = [scores[j] if j not in tried else -1e9 for j in range(n_cat)]
            J = int(np.argmax(cand_scores))

            if J in tried or all(s < -1e8 for s in cand_scores):
                J_best = int(np.argmax(scores))
                W[:, J_best] = (1-0.3) * W[:, J_best] + 0.3 * I
                break

            rho = scores[J] / (np.sum(I) + 1e-12)
            if rho >= vigilance:
                W[:, J] = (1-0.3) * W[:, J] + 0.3 * I
                break
            else:
                tried.add(J)
                W[:, J] = np.zeros(n_feat*2)

    return W

# Test data
X_art = np.array([[1,0,0],[0,1,0],[1,1,0],[0,0,1]])
print("ART1 categories:", art1_train(X_art))
print("ART2 categories:", art2_train(X_art))
