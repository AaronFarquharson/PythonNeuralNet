# Works,thank goodness!

import numpy as np

class NeuralNetwork:
    def __init__(self,inputSize,hiddenSize,outputSize,lr):
        self.w1=np.random.rand(inputSize,hiddenSize)
        self.w2=np.random.rand(hiddenSize,outputSize)
        self.b1=np.random.rand(1,hiddenSize)
        self.b2=np.random.rand(1,outputSize)
        self.lr=lr

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self,input):
        self.z1=np.dot(input,self.w1)+self.b1
        self.a1=self.sigmoid(self.z1)
        self.z2=np.dot(self.a1,self.w2)+self.b2
        return self.sigmoid(self.z2)

    def backprop(self,input,desiredOutput):
        output=self.feedforward(input)
        # application of the chain rule to find derivative of the loss function with respect to w2 and w1
        d_output=(desiredOutput - output)*self.sigmoid_derivative(self.z2)
        d_hidden1=(np.dot(d_output,self.w2.T)*self.sigmoid_derivative(self.z1))


        d_w2=np.dot(self.a1.T,d_output)
        d_w1=np.dot(input.T,d_hidden1)
        d_b2=np.sum(d_output,axis=0,keepdims=True)
        d_b1=np.sum(d_hidden1,axis=0,keepdims=True)


        # update the weights with the derivative (slope) of the loss function
        self.w1+=self.lr*d_w1
        self.w2+=self.lr*d_w2
        self.b1+=self.lr*d_b1
        self.b2+=self.lr*d_b2

    def trainNetwork(self,input,output,epochs):
        for i in range(epochs):
            [input,output]=unison_shuffled_copies(input,output)
            self.backprop(input,output)


def main():
    input=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,0,0],[0,0,0]])
    output=np.array([[0],[1],[1],[0],[1],[0]])
    net=NeuralNetwork(input.shape[1],4,output.shape[1],0.1)
    net.trainNetwork(input,output,10000)
    input=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    print(net.feedforward(input))

def unison_shuffled_copies(a, b):
    assert len(a)==len(b)
    p=np.random.permutation(len(a))
    return a[p], b[p]

if __name__=="__main__":
    main()
