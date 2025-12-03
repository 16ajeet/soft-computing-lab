# 13. Write a Python program to implement ANN with Backpropagation Learning Algorithm.
# 14. Write a program for XOR gate using backpropagation (3 binary inputs → 1 output). Use
# sigmoid activation, learning rate=0.1, epochs=10. Draw error vs epoch graph.
# 15. Build ANN with 5 input features, 3 hidden layers (10,7,5 neurons), and 1 output neuron.
# Generate random dataset (100 samples). Target: binary (sum(features)>2.5). Train with
# sigmoid activation, learning rate=0.1, epochs=100. Draw error vs epoch graph.

import math

def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))
def sig_deriv(o): return o*(1-o)

class ANN:
    def __init__(self):
        self.w1 = [[0.1]*2 for _ in range(4)]  # 2->4 hidden
        self.b1 = [0]*4
        self.w2 = [[0.1] for _ in range(4)]    # 4->1 output
        self.b2 = [0]
    
    def forward(self, x):
        self.h = [sigmoid(sum(x[j]*self.w1[i][j] for j in range(2))+self.b1[i]) for i in range(4)]
        self.o = sigmoid(sum(self.h[i]*self.w2[i][0] for i in range(4))+self.b2[0])
        return self.o
    
    def train(self, X, y, lr=0.1, epochs=1000):
        for _ in range(epochs):
            for xi, t in zip(X, y):
                o = self.forward(xi)
                e = t - o
                
                # Output layer deltas
                d2 = e * sig_deriv(o)
                for i in range(4):
                    self.w2[i][0] += lr * d2 * self.h[i]
                self.b2[0] += lr * d2
                
                # Hidden layer deltas
                for i in range(4):
                    d1 = d2 * self.w2[i][0] * sig_deriv(self.h[i])
                    for j in range(2):
                        self.w1[i][j] += lr * d1 * xi[j]
                    self.b1[i] += lr * d1

# XOR test
net = ANN()
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]
net.train(X, y)
for x in X:
    print(f"{x} -> {net.forward(x):.3f}")


#==============================================================
import matplotlib.pyplot as plt
import math

def sigmoid(x): return 1/(1+math.exp(-x))
def sig_deriv(o): return o*(1-o)

X = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
y = [0,1,1,0,1,0,0,1]  # 3-input XOR truth table

w1 = [[0.1]*3 for _ in range(4)]
b1 = [0]*4
w2 = [[0.1] for _ in range(4)]
b2 = [0]

errors = []
lr = 0.1

for epoch in range(10):
    total_error = 0
    for xi, t in zip(X, y):
        # Forward
        h = [sigmoid(sum(xi[j]*w1[i][j] for j in range(3))+b1[i]) for i in range(4)]
        o = sigmoid(sum(h[i]*w2[i][0] for i in range(4))+b2[0])
        
        # Backward
        e = t - o
        total_error += 0.5 * e**2
        
        d2 = e * sig_deriv(o)
        for i in range(4):
            w2[i][0] += lr * d2 * h[i]
        b2[0] += lr * d2
        
        for i in range(4):
            d1 = d2 * w2[i][0] * sig_deriv(h[i])
            for j in range(3):
                w1[i][j] += lr * d1 * xi[j]
            b1[i] += lr * d1
    
    errors.append(total_error)
    print(f"Epoch {epoch+1}: Error = {total_error:.4f}")

plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('XOR Backpropagation Error vs Epoch')
plt.savefig('XOR Backpropagation Error vs Epoch')
plt.show()


#================================================================================
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-500,500)))
def sig_deriv(o): return o*(1-o)

# Generate dataset
np.random.seed(42)
X = np.random.rand(100,5)
y = (np.sum(X, axis=1) > 2.5).astype(float)

# Network: 5->10->7->5->1
w1 = np.random.randn(10,5)*0.1; b1 = np.zeros(10)
w2 = np.random.randn(7,10)*0.1; b2 = np.zeros(7)
w3 = np.random.randn(5,7)*0.1; b3 = np.zeros(5)
w4 = np.random.randn(1,5)*0.1; b4 = np.zeros(1)

lr = 0.1
errors = []

for epoch in range(100):
    total_error = 0
    for xi, t in zip(X, y):
        # Forward
        h1 = sigmoid(np.dot(w1, xi) + b1)
        h2 = sigmoid(np.dot(w2, h1) + b2)
        h3 = sigmoid(np.dot(w3, h2) + b3)
        o = sigmoid(np.dot(w4, h3) + b4)
        
        # Backward
        e = t - o
        total_error += 0.5 * e**2
        
        d4 = e * sig_deriv(o)
        w4 += lr * np.outer(d4, h3)
        b4 += lr * d4
        
        d3 = np.dot(w4.T, d4) * sig_deriv(h3)
        w3 += lr * np.outer(d3, h2)
        b3 += lr * d3
        
        d2 = np.dot(w3.T, d3) * sig_deriv(h2)
        w2 += lr * np.outer(d2, h1)
        b2 += lr * d2
        
        d1 = np.dot(w2.T, d2) * sig_deriv(h1)
        w1 += lr * np.outer(d1, xi)
        b1 += lr * d1
    
    errors.append(total_error)

plt.plot(errors)
plt.xlabel('Epochs'); plt.ylabel('Error')
plt.title('Deep ANN Error vs Epoch')
plt.savefig('Deep ANN Error vs Epoch')
plt.show()


#==============================================
