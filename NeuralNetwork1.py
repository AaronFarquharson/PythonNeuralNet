import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)
        self.lr = 0.1

    def sigmoid(self, vals):
        return 1/(1 + np.exp(-vals))

    def sigmoid_derivative(self, vals):
        return self.sigmoid(vals)*(1-self.sigmoid(vals))

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def printOutputs(self):
        print(self.output)

def main():
    input = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    output = np.array([[0],[1],[1],[0]])
    net = NeuralNetwork(input, output)
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
    for i in range(500):
        net.feedforward()
        net.backprop()
    net.printOutputs()
    for i in range(500):
        net.feedforward()
        net.backprop()
    net.printOutputs()


if __name__ == "__main__":
    main()
