# 5. Implement ANN model using XOR gate with two layer architecture for given x1 and
# x2. Weights are bipolar {-1, 1}. Find all possible weight combinations that satisfy
# XOR.
# 6. Implement ANN model for two-level OR-AND implementation with inputs x1 and x2
# (bipolar weights).
# 7. Implement NAND gate ANN model by adding bias b=1. Calculate three weights w0,
# w1, and w2.
# 8. Design a Hebb net to implement OR and AND functions (consider bipolar inputs and
# targets).

def sign(x): return 1 if x >= 0 else -1

inputs = [(-1,-1), (-1,1), (1,-1), (1,1)]
targets = [-1, 1, 1, -1]

solutions = []

for w_ih in [(a,b,c,d) for a in [-1,1] for b in [-1,1] for c in [-1,1] for d in [-1,1]]:
 for b_h in [(e,f) for e in [-1,1] for f in [-1,1]]:
  for w_ho in [(g,h) for g in [-1,1] for h in [-1,1]]:
   for b_o in [-1,1]:
    ok = True
    for x,t in zip(inputs, targets):
     h = [sign(w_ih[0]*x[0]+w_ih[1]*x[1]+b_h[0]), sign(w_ih[2]*x[0]+w_ih[3]*x[1]+b_h[1])]
     o = sign(w_ho[0]*h[0] + w_ho[1]*h[1] + b_o)
     if o != t:
      ok = False
      break
    if ok:
     solutions.append((w_ih, b_h, w_ho, b_o))

print(f"Solutions: {len(solutions)}")
for sol in solutions:
 print(sol)

#6.========================================
def sign(x): return 1 if x >= 0 else -1

def forward(x1, x2):
    # Level 1: OR neuron (w=[1,1], b=-0.5)
    h_or = sign(1*x1 + 1*x2 - 0.5)
    # Level 2: AND neuron (w=1, b=-0.5)  
    out = sign(1*h_or - 0.5)
    return out

# Test cases
print("x1 x2 | OR-AND")
for x1 in [-1,1]:
    for x2 in [-1,1]:
        print(f"{x1:2} {x2:2} |  {forward(x1,x2):2}")

#7.===================================================
def nand(x1, x2):
    w0, w1, w2 = 1, -1, -1  # bias=1, weights calculated
    s = w0 + w1*x1 + w2*x2
    return 1 if s >= 0 else -1

print("x1 x2 | NAND (bipolar)")
for x1 in [-1,1]:
    for x2 in [-1,1]:
        print(f"{x1:2} {x2:2} |  {nand(x1,x2):2}")

#8.================================================
def hebb(p, t, epochs=5):
    w = [0, 0]
    for _ in range(epochs):
        for x, y in zip(p, t):
            act = 1 if sum(xi*wi for xi, wi in zip(x, w)) >= 0 else -1
            if act == y:
                w = [wi + x[i]*y for i, wi in enumerate(w)]
    return w

# OR: [-1,-1]->-1, others->1
p_or = [[-1,-1],[-1,1],[1,-1],[1,1]]
t_or = [-1,1,1,1]
w_or = hebb(p_or, t_or)

# AND: [1,1]->1, others->-1
p_and = [[-1,-1],[-1,1],[1,-1],[1,1]]
t_and = [-1,-1,-1,1]
w_and = hebb(p_and, t_and)

print("OR weights:", w_or)
print("AND weights:", w_and)
