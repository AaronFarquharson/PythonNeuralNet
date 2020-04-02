# Works, thank goodness!

import numpy as np

class NeuralNetwork:
    def __init__(self, x, y, lr):
        self.input = x
        self.w1 = np.random.rand(self.input.shape[1], 4)
        self.w2 = np.random.rand(4, 1)
        self.b1 = np.random.rand(1,4)
        self.b2 = np.random.random()
        self.y = y
        self.output = np.zeros(y.shape)
        self.lr = lr

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self):
        self.z1 = np.dot(self.input, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.output = self.sigmoid(self.z2)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to w2 and w1
        d_output = (self.y - self.output) * self.sigmoid_derivative(self.z2)
        d_hidden1 = (np.dot(d_output, self.w2.T) * self.sigmoid_derivative(self.z1))


        d_w2 = np.dot(self.a1.T, d_output)
        d_w1 = np.dot(self.input.T, d_hidden1)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)
        d_b1 = np.sum(d_hidden1, axis=0, keepdims=True)

        # update the weights with the derivative (slope) of the loss function
        self.w1 += self.lr * d_w1
        self.w2 += self.lr * d_w2
        self.b1 += self.lr * d_b1
        self.b2 += self.lr * d_b2

    def printOutputs(self):
        print(self.output)

    def evaluate(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        return self.sigmoid(z2)


def main():
    input = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    output = np.array([[0],[1],[1],[0]])
    net = NeuralNetwork(input, output, 1)
    for i in range(500):
        net.feedforward()
        net.backprop()
    net.printOutputs()
    for i in range(500):
        net.feedforward()
        net.backprop()
    net.printOutputs()
    for i in range(500):
        net.feedforward()
        net.backprop()
    net.printOutputs()

    input = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    print(net.evaluate(input))

if __name__ == "__main__":
    main()
