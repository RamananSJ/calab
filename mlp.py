import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.4
w6 = 0.45
w7 = 0.5
w8 = 0.55
b1=0.35
b2=0.6
learning_rate = 0.8

# Inputs
input1 = 0.05
input2 = 0.10

# Target outputs
t1 = 0.01
t2 = 0.99

# Training iterations
num_iterations = 1


# Training loop
for iteration in range(num_iterations):
    # Forward Pass
    z_h1 = w1 * input1 + w3 * input2 +b1
    z_h2 = w2 * input1 + w4 * input2 +b1
    h1 = sigmoid(z_h1)
    h2 = sigmoid(z_h2)

    z_o1 = w5 * h1 + w7 * h2 +b2
    z_o2 = w6 * h1 + w8 * h2 +b2
    o1 = sigmoid(z_o1)
    o2 = sigmoid(z_o2)

    # Calculate errors


    # Backward Pass
    delta_o1 = -(t1 - o1) * o1 * (1 - o1)
    delta_o2 = -(t2 - o2) * o2 * (1 - o2)

    delta_h1 = (delta_o1 * w5 + delta_o2 * w6) * h1 * (1 - h1)
    delta_h2 = (delta_o1 * w7 + delta_o2 * w8) * h2 * (1 - h2)

    # Update weights and biases
    w5 -= learning_rate * delta_o1 * h1
    w6 -= learning_rate * delta_o2 * h1
    w7 -= learning_rate * delta_o1 * h2
    w8 -= learning_rate * delta_o2 * h2

    w1 -= learning_rate * delta_h1 * input1
    w2 -= learning_rate * delta_h2 * input1
    w3 -= learning_rate * delta_h1 * input2
    w4 -= learning_rate * delta_h2 * input2
E_o1 = 0.5 * (t1 - o1) ** 2
E_o2 = 0.5 * (t2 - o2) ** 2
print(E_o1,E_o2)
print(o1,o2)
# Print the final weights and biases
print("Final Weights and Biases:")
print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, w3 = {w3:.4f}, w4 = {w4:.4f}")
print(f"w5 = {w5:.4f}, w6 = {w6:.4f}, w7 = {w7:.4f}, w8 = {w8:.4f}")
