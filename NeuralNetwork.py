from Matrix import Matrix
import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork():
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = []
        self.biases = []
        for i in range(1,len(sizes)):
            data = NeuralNetwork.__random_data(sizes[i],sizes[i-1])
            self.weights.append(Matrix(sizes[i], sizes[i-1], data))
            data = [[1]]*sizes[i]
            self.biases.append(Matrix(sizes[i],1,data))

    def __random_data(n, m):
        width = 1/math.sqrt(n)
        return [
            [random.uniform(-width,width) for i in range(m)]
            for j in range(n)
        ]

    def feedforword(self, inputs):
        if len(inputs) == self.sizes[0]:
            output = Matrix(len(inputs),1, inputs)
            for i in range(len(self.sizes)-1):
                output = self.weights[i]*output+self.biases[i]
                output = output.apply(sigmoid)
            return output
        raise Exception("Invalid input length")
        
    def backpropagation(self, x, y
        """
        delta_new:

                weight.transpose()
                *
                delta
            **
                output.apply(dsigmoid)
        *
            sigmoided.transpose()
        """
        output = x
        layers = [x]
        for weight, bias in zip(self.weights,self.biases):
            output = weight*output+bias
            layers.append(output)
            output = output.apply(sigmoid)
            layers.append(output)

        delta_w, delta_b  = [], []
        sigmoided, output  = layers.pop(), layers.pop()
        delta = (sigmoided-y)**(output.apply(dsigmoid))
        
        for weight in reversed(self.weights):
            sigmoided = layers.pop()
            delta_b.insert(0,delta)
            delta_w.insert(0,delta*(sigmoided.transpose()))
            if layers:
                output = layers.pop()
                delta = (weight.transpose()*delta)**(output.apply(dsigmoid))
        return (delta_w,delta_b)

    def update(self, batch):
        for x,y in batch:
            delta_w, delta_b = self.backpropagation(x, y)
            for i, w, b in zip(range(len(self.sizes)),delta_w, delta_b):
                self.weights[i] -= w
                self.biases[i]  -= b
                    
    def train(self, batch):
        for i in range(5000):
            random.shuffle(batch)
            self.update(batch)

if __name__ == "__main__":
    nn       = NeuralNetwork([2,2,1])
    print("Not traind")
    print(nn.feedforword([[0],[0]]))
    print(nn.feedforword([[0],[1]]))
    print(nn.feedforword([[1],[0]]))
    print(nn.feedforword([[1],[1]]))
    
    batch = [(Matrix(2,1,[[0],[0]]),Matrix(1,1,[[0]]))]
    batch.append((Matrix(2,1,[[0],[1]]),Matrix(1,1,[[1]])))
    batch.append((Matrix(2,1,[[1],[0]]),Matrix(1,1,[[1]])))
    batch.append((Matrix(2,1,[[1],[1]]),Matrix(1,1,[[0]])))
    nn.update(batch)
    
    print("Traind")
    print(nn.feedforword([[0],[0]]))
    print(nn.feedforword([[0],[1]]))
    print(nn.feedforword([[1],[0]]))
    print(nn.feedforword([[1],[1]]))