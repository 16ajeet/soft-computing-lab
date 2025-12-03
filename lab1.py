# 1. Write a Python program to implement McCulloch-Pitts neuron for basic logic gates
# (AND, OR, NOT, NAND, NOR).
# 2. Write a Python program for verification of XOR gate using McCulloch-Pitts neuron.
# 3. Write a Python program to implement Hebb Network (without inbuilt functions).
# 4. Write a Python program to implement Multilayer Perceptron (without inbuilt
# functions).

# No imports needed
def s(x):return 1 if x>=0 else 0
def mp(i,w,b):return s(sum(x*w for x,w in zip(i,w))+b)

def AND(i):return mp(i,[1,1],-1.5)
def OR(i):return mp(i,[1,1],-0.5)
def NAND(i):return mp(i,[-1,-1],1.5)
def NOR(i):return mp(i,[-1,-1],0.5)
def NOT(x):return mp([x],[-1],0.5)

for x1 in[0,1]:
 for x2 in[0,1]:
  print(f"{x1}{x2}: AND={AND([x1,x2])},OR={OR([x1,x2])},NAND={NAND([x1,x2])},NOR={NOR([x1,x2])},NOT={NOT(x1)}")

#========================================================
# No imports needed
def s(x):return 1 if x>=0 else 0
def mp(i,w,b):return s(sum(x*w for x,w in zip(i,w))+b)

for w1 in[-2,-1,0,1,2]:
 for w2 in[-2,-1,0,1,2]:
  for b in[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]:
   c=0
   for x1 in[0,1]:
    for x2 in[0,1]:
     c+=(mp([x1,x2],[w1,w2],b)==(x1^x2))
   if c==4:print(f"XOR works: w=[{w1},{w2}], b={b}")
print("No solution = Single neuron can't do XOR!")

#==============================================
# No imports needed
def hebb(p,t,e=3):
 w=[0]*2
 for _ in range(e):
  for x,y in zip(p,t):
   if sum(xi*wi for xi,wi in zip(x,w))>=0==y:
    w=[wi+x[i]*y for i,wi in enumerate(w)]
 return w

p=[[1,0,0],[0,0,0]];t=[1,0]
print("Hebb weights:",hebb(p,t))

#===========================================
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]

# Initialize weights and biases
w1 = [[0.1, 0.2], [0.3, 0.4]]
b1 = [0, 0]
w2 = [0.5, 0.6]
b2 = 0
lr = 0.5

for _ in range(1000):
    for i, target in enumerate(y):
        # Forward pass
        hidden = [sigmoid(sum(X[i][j]*w1[k][j] for j in range(2)) + b1[k]) for k in range(2)]
        output = sigmoid(sum(hidden[j]*w2[j] for j in range(2)) + b2)
        
        # Error
        error = target - output
        
        # Backpropagation output layer
        delta_output = error * output * (1 - output)
        
        # Update output weights and bias
        for j in range(2):
            w2[j] += lr * delta_output * hidden[j]
        b2 += lr * delta_output
        
        # Backpropagation hidden layer
        for k in range(2):
            delta_hidden = delta_output * w2[k] * hidden[k] * (1 - hidden[k])
            for j in range(2):
                w1[k][j] += lr * delta_hidden * X[i][j]
            b1[k] += lr * delta_hidden

# Testing
for x in X:
    hidden = [sigmoid(sum(x[j]*w1[k][j] for j in range(2)) + b1[k]) for k in range(2)]
    output = sigmoid(sum(hidden[j]*w2[j] for j in range(2)) + b2)
    print(f"Input: {x} Output: {output:.3f}")
