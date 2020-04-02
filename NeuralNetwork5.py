# Works,thank goodness!

import numpy as np

class NeuralNetwork:
    def __init__(self,inputSize,outputSize,lr,hiddenLayers=None):
        # Initialize input layer
        self.weights = [np.random.rand(inputSize,hiddenLayers[0])]
        self.biases = [np.random.rand(1,hiddenLayers[0])]
        # Initialize hidden layers
        if len(hiddenLayers) > 1:
            for i in range(1,len(hiddenLayers)):
                self.weights.append(np.random.rand(hiddenLayers[i-1],hiddenLayers[i]))
                self.biases.append(np.random.rand(1,hiddenLayers[i]))
        # Initialize output layer
        self.weights.append(np.random.rand(hiddenLayers[len(hiddenLayers)-1],outputSize))
        self.biases.append(np.random.rand(1,outputSize))
        self.lr=lr

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self,input):
        self.z = [np.dot(input,self.weights[0])+self.biases[0]]
        self.a = [self.sigmoid(self.z[0])]
        for i in range(1,len(self.weights)):
            self.z.append(np.dot(self.a[i-1],self.weights[i])+self.biases[i])
            self.a.append(self.sigmoid(self.z[i]))
        return self.a[len(self.a)-1]

    def backprop(self,input,desiredOutput):
        output=self.feedforward(input)
        # application of the chain rule to find derivative of the loss function with respect to w2 and w1
        d_layer=[(desiredOutput - output)*self.sigmoid_derivative(self.z[len(self.z)-1])]
        for i in range(1,len(self.weights)):
            d_layer.append(np.dot(d_layer[i-1],self.weights[len(self.weights)-i].T)*self.sigmoid_derivative(self.z[len(self.z)-i-1]))
        # Put the layer derivatives in a more logical order
        d_layer.reverse()


        d_weights=[np.dot(input.T,d_layer[0])]
        for i in range(1,len(d_layer)):
            d_weights.append(np.dot(self.a[i-1].T,d_layer[i]))

        d_biases=[np.sum(d_layer[0],axis=0,keepdims=True)]
        for i in range(1,len(d_layer)):
            d_biases.append(np.sum(d_layer[i],axis=0,keepdims=True))


        # update the weights with the derivative (slope) of the loss function
        for i in range(len(self.weights)):
            self.weights[i] += self.lr*d_weights[i]
        for i in range(len(self.biases)):
            self.biases[i] += self.lr*d_biases[i]

    def trainNetwork(self,input,output,epochs):
        for i in range(epochs):
            [input,output]=unison_shuffled_copies(input,output)
            self.backprop(input,output)


def main():
    input=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,0,0],[0,0,0]])
    output=np.array([[0],[1],[1],[0],[1],[0]])
    net=NeuralNetwork(input.shape[1],output.shape[1],0.1,[4,3])
    net.trainNetwork(input,output,10000)
    input=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    print(net.feedforward(input))

def unison_shuffled_copies(a, b):
    assert len(a)==len(b)
    p=np.random.permutation(len(a))
    return a[p], b[p]

if __name__=="__main__":
    main()
