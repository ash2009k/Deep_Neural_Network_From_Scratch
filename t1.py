# Imports
import numpy as np
from matplotlib import pyplot as plt
import pickle
from mpl_toolkits import mplot3d

x = np.loadtxt('train100.txt')
y = x[..., 2]
X = x[..., :2]
X11=x[..., 0]
X22=x[..., 1]


val_x = np.loadtxt('val.txt')
val_y = val_x[..., 2]
val_X = val_x[..., :2]

eta = 0.000001
alpha = 0.9


# Activation function
def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

# Class definition
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 10)
        self.weights2 = np.random.rand(10, 8)
        self.weights3 = np.random.rand(8, 1)

        self.bias1 = np.zeros((1, 10))
        self.bias2 = np.zeros((1, 8))
        self.bias3 = np.zeros((1, 1))

        self.past_weights1 = np.zeros((self.input.shape[1], 10))
        self.past_weights2 = np.zeros((10, 8))
        self.past_weights3 = np.zeros((8, 1))

        self.past_bias1 = np.zeros((1, 10))
        self.past_bias2 = np.zeros((1, 8))
        self.past_bias3 = np.zeros((1, 1))
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = tanh(np.dot(self.input, self.weights1) + self.bias1)
        self.layer2 = tanh(np.dot(self.layer1, self.weights2) + self.bias2)
        self.layer3 = (np.dot(self.layer2, self.weights3) + self.bias3)

        return self.layer3

    def backprop(self):
        del3 = (self.y - self.output)
        d_weights3 = np.dot(del3, self.layer2) * eta + self.past_weights3.T * alpha
        d_bias3=del3.T * eta+self.past_bias3*alpha

        del2 = np.dot(self.weights3, del3) * tanh_derivative(np.dot(self.layer1, self.weights2) + self.bias2).T
        d_weights2 = np.dot(del2, self.layer1) * eta + self.past_weights2.T * alpha
        d_bias2 = del2.T * eta + self.past_bias2 * alpha

        del1 = np.dot(self.weights2, del2) * tanh_derivative(np.dot(self.input, self.weights1) + self.bias1).T
        d_weights1 = np.dot(del1, [self.input]) * eta + self.past_weights1.T * alpha
        d_bias1 = del1.T * eta + self.past_bias1 * alpha

        self.weights1 += d_weights1.T
        self.weights2 += d_weights2.T
        self.weights3 += d_weights3.T
        self.bias1 += d_bias1
        self.bias2 += d_bias2
        self.bias3 += d_bias3
        self.past_weights1 = d_weights1.T
        self.past_weights2 = d_weights2.T
        self.past_weights3 = d_weights3.T
        self.past_bias1=d_bias1
        self.past_bias2 = d_bias2
        self.past_bias3 = d_bias3
    def result(self,X):
        self.layer1 = tanh(np.dot(X, self.weights1) + self.bias1)
        self.layer2 = tanh(np.dot(self.layer1, self.weights2) + self.bias2)
        self.layer3 = (np.dot(self.layer2, self.weights3) + self.bias3)
        return self.layer3
    def train(self, X, y):
        self.input = X
        self.y = y
        self.output = np.zeros(y.shape)
        self.output = self.feedforward()
        self.backprop()


NN = NeuralNetwork(X, y)
epoch=[]
train_error=[]
for i in range(10000):  # trains the NN 1,000 times
    for j in range(len(y)):
        NN.train(np.array(X[j]), np.array(y[j]))
        if i == 999:
            print("Input : \n" + str(X[j]))
            print("Actual Output: \n" + str(y[j]))
            print("Predicted Output: \n" + str(NN.output))
            print("\n")
    val_loss=np.mean(np.square(np.atleast_2d(val_y).T - NN.result(val_X)))
    print('val_loss '+str(val_loss))
    epoch.append(i)
    train_error.append(np.mean(np.square(np.atleast_2d(y).T - NN.result(X))))
    if(i>1 and (train_error[i-1]-train_error[i])<0.000001):
        break
    print("Loss: \n" + str(np.mean(np.square(np.atleast_2d(y).T - NN.result(X)))))  # mean sum squared loss
    #print("Loss: \n" + str(np.mean(np.square(y.T - o))))  # mean sum squared loss


with open('as1_NN', 'wb') as output:
    pickle.dump(NN, output, pickle.HIGHEST_PROTOCOL)


plt.title('Q1.1_plot')
plt.xlabel('epochs')
plt.ylabel('Error')
plt.plot(epoch, train_error)
plt.show()

plt.title('Ass1_Q1.2_plot')
plt.xlabel('epochs')
plt.ylabel('Error')
plt.scatter(np.atleast_2d(val_y).T , NN.result(val_X))
plt.plot([-20,140],[-20,140],color='red')
plt.show()

x1 =np.arange(-20,140,1)
x2 =np.arange(-20,140,1)
XX, YY = np.meshgrid(x1, x2)
o=[]
for i in x1:
    for j in x2:
        NN.result([[i,j]])
        o.append(NN.layer3)
ax = plt.axes(projection='3d')
ax.set_title('epoch' + str(epoch) + 'H_1' + 'node' + str(i))
#ax.contour3D(XX, YY, np.reshape(o, (160, 160)), 50, cmap='binary')
#ax.plot_surface(XX, YY, np.reshape(o, (160, 160)), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.plot_wireframe(XX, YY, np.reshape(o, (160, 160)), color='black')
ax.scatter3D(X11, X22, y);

print(y.shape)
plt.show()
val_loss=np.mean(np.square(np.atleast_2d(val_y).T - NN.result(val_X)))
print('val_loss '+str(val_loss))
#print(NN.result(X))
print(np.atleast_2d(val_y).T - NN.result(val_X))


