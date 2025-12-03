# 9. Write a program for solving linearly separable problem using Perceptron Model (AND
# Gate with bipolar inputs and outputs). Train with learning rates {0.1, 0.01, 0.001}. Draw
# graph (loss vs iterations).
# 10. Write a program for solving non-linearly separable problem using Perceptron Model. Train
# with learning rates {0.1, 0.01, 0.001}.
# 11. Write a program for solving linearly separable problem using Adaline Model (AND Gate,
# bipolar). Train with learning rates {0.1, 0.01, 0.001}.
# 12. Write a program for solving non-linearly separable problem using Madaline Model (AND
# # Gate, bipolar). Train with learning rates {0.1, 0.01, 0.001}.

#9.
import matplotlib.pyplot as plt

def sign(x):
    return 1 if x >= 0 else -1

def train_perceptron(X, y, lr, epochs=50):
    w = [0, 0]
    b = 0
    losses = []
    for _ in range(epochs):
        total_loss = 0
        for xi, target in zip(X, y):
            net = sum(x * wi for x, wi in zip(xi, w)) + b
            pred = sign(net)
            error = target - pred
            if error != 0:
                w = [wi + lr * error * x for wi, x in zip(w, xi)]
                b += lr * error
            total_loss += abs(error)
        losses.append(total_loss)
    return w, b, losses

X = [[1,1], [1,-1], [-1,1], [-1,-1]]
y = [1, -1, -1, -1]

for lr in [0.1, 0.01, 0.001]:
    w, b, losses = train_perceptron(X, y, lr)
    print(f"Learning rate: {lr}")
    print(f"Weights: {w}, Bias: {b}")
    plt.plot(losses, label=f'lr={lr}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs for AND gate Perceptron')
plt.savefig('Loss vs Epochs for AND gate Perceptron')
plt.show()


#10.======================================
import matplotlib.pyplot as plt

def sign(x):
    return 1 if x >= 0 else -1

def train_perceptron(X, y, lr, epochs=50):
    w = [0, 0]
    b = 0
    losses = []
    for _ in range(epochs):
        total_loss = 0
        for xi, target in zip(X, y):
            net = sum(x * wi for x, wi in zip(xi, w)) + b
            pred = sign(net)
            error = target - pred
            if error != 0:
                w = [wi + lr * error * x for wi, x in zip(w, xi)]
                b += lr * error
            total_loss += abs(error)
        losses.append(total_loss)
    return w, b, losses

# XOR bipolar inputs/outputs - non-linearly separable
X = [[1,1], [1,-1], [-1,1], [-1,-1]]
y = [-1, 1, 1, -1]

for lr in [0.1, 0.01, 0.001]:
    w, b, losses = train_perceptron(X, y, lr)
    print(f"Learning rate: {lr}")
    print(f"Weights: {w}, Bias: {b}")
    plt.plot(losses, label=f'lr={lr}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs for non-linearly separable problem (XOR)')
plt.savefig('Loss vs Epochs for non-linearly separable problem (XOR)')
plt.show()


#11.=============================================================
import matplotlib.pyplot as plt

def adaline_train(X, y, lr, epochs=50):
    w = [0, 0]
    b = 0
    losses = []
    for _ in range(epochs):
        total_loss = 0
        for xi, target in zip(X, y):
            net = sum(x * wi for x, wi in zip(xi, w)) + b
            error = target - net  # Adaline uses linear activation
            w = [wi + lr * error * x for wi, x in zip(w, xi)]
            b += lr * error
            total_loss += error ** 2
        losses.append(total_loss)
    return w, b, losses

X = [[1,1], [1,-1], [-1,1], [-1,-1]]
y = [1, -1, -1, -1]

for lr in [0.1, 0.01, 0.001]:
    w, b, losses = adaline_train(X, y, lr)
    print(f"Learning rate: {lr}")
    print(f"Weights: {w}, Bias: {b}")
    plt.plot(losses, label=f'lr={lr}')

plt.xlabel('Epochs')
plt.ylabel('Squared Error Loss')
plt.legend()
plt.title('Adaline Training for AND gate (bipolar)')
plt.savefig('Adaline Training for AND gate (bipolar)')
plt.show()


#12.========================================================
import matplotlib.pyplot as plt

def madaline_train(X, y, lr, epochs=50):
    # 2 hidden Adalines + 1 output Adaline
    w_hidden1 = [0, 0]; b_hidden1 = 0
    w_hidden2 = [0, 0]; b_hidden2 = 0
    w_output = [0, 0]; b_output = 0
    losses = []
    
    for _ in range(epochs):
        total_loss = 0
        for xi, target in zip(X, y):
            # Hidden layer (linear activation)
            h1 = sum(xi[j]*w_hidden1[j] for j in range(2)) + b_hidden1
            h2 = sum(xi[j]*w_hidden2[j] for j in range(2)) + b_hidden2
            
            # Output layer
            net_out = w_output[0]*h1 + w_output[1]*h2 + b_output
            error = target - net_out
            
            # Update output weights
            w_output[0] += lr * error * h1
            w_output[1] += lr * error * h2
            b_output += lr * error
            
            # Update hidden weights (gradient through output)
            dh1 = error * w_output[0]
            dh2 = error * w_output[1]
            for j in range(2):
                w_hidden1[j] += lr * dh1 * xi[j]
                w_hidden2[j] += lr * dh2 * xi[j]
            b_hidden1 += lr * dh1
            b_hidden2 += lr * dh2
            
            total_loss += error ** 2
        losses.append(total_loss)
    return losses

X = [[1,1], [1,-1], [-1,1], [-1,-1]]
y = [1, -1, -1, -1]  # AND gate

for lr in [0.1, 0.01, 0.001]:
    losses = madaline_train(X, y, lr)
    plt.plot(losses, label=f'lr={lr}')

plt.xlabel('Epochs')
plt.ylabel('Squared Error Loss')
plt.legend()
plt.title('Madaline Training for AND gate (bipolar)')
plt.savefig('Madaline Training for AND gate (bipolar)')
plt.show()
