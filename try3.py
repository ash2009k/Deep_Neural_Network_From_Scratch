# Imports
import numpy as np

x = np.loadtxt('train100.txt')
y = x[..., 2]
X = x[..., :2]
eta = 0.001
alpha = 0.8


# Activation function
def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


# number of nodes in layers
n = [2, 4, 5, 1]

weights1 = np.random.rand(n[0], n[1])
weights2 = np.random.rand(n[1], n[2])
weights3 = np.random.rand(n[2], n[3])
bias1 = np.random.rand(n[1], 1)
bias2 = np.random.rand(n[2], 1)
bias3 = np.random.rand(n[3], 1)
out1 = np.zeros((n[1], 1))
out2 = np.zeros((n[2], 1))
out3 = np.zeros((n[3], 1))
del1 = np.zeros((n[1], 1))
del2 = np.zeros((n[2], 1))
del3 = np.zeros((n[3], 1))

for epochs in range(1):
    for datapoints in range(len(y)):
        # feed forward
        input = X[datapoints]
        output = y[datapoints]
        out1 = tanh(np.dot(input, weights1) + bias1.T)
        out2 = tanh(np.dot(out1, weights2) + bias2.T)
        out3 = np.dot(out2, weights3) + bias3.T
        # feedforward done

        # backpropagation
        del3 = (out3 - output)
        d_weights3 = np.dot(del3, self.layer2) * eta + self.past_weights3.T * alpha

        del2 = np.dot(self.weights3, del3) * tanh_derivative(np.dot(self.layer1, self.weights2) + self.bias2).T
        d_weights2 = np.dot(del2, self.layer1) * eta + self.past_weights2.T * alpha

        del1 = np.dot(self.weights2, del2) * tanh_derivative(np.dot(self.input, self.weights1) + self.bias1).T
        d_weights1 = np.dot(del1, [self.input]) * eta + self.past_weights1.T * alpha


