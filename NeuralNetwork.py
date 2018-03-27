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
        
    def backpropagation(self, x, y):
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

    def update_mini_batch(self, batch, lr):
        mean_w = [w.apply(lambda x: 0) for w in self.weights]
        mean_b = [b.apply(lambda x: 0) for b in self.biases]

        # summing all the deltas
        for x,y in batch:
            delta_w, delta_b = self.backpropagation(x, y)
            for i in range(len(self.sizes)-1):
                mean_w[i] += delta_w[i]
                mean_b[i] += delta_b[i]

        # averaging all the deltas
        for i in range(len(self.sizes)-1):
            mean_w[i] = mean_w[i].apply(lambda x: lr*x/len(batch))
            mean_b[i] = mean_b[i].apply(lambda x: lr*x/len(batch))
    
        # subtract the estimated deltas
        for i, w, b in zip(range(len(self.sizes)),mean_w, mean_b):
            self.weights[i] -= w
            self.biases[i]  -= b
                    
    def train(self, data, lr, epochs, mini_batch_size=0):
        if mini_batch_size < 1:
            mini_batch_size = len(data)
        for i in range(epochs):
            random.shuffle(data)
            mini_batches = [
                data[k:k+mini_batch_size]
                for k in range(0, len(data), mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            

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
    nn.train(batch, lr=2, epochs=5000, mini_batch_size=2)
    
    print("Traind")
    print(nn.feedforword([[0],[0]]))
    print(nn.feedforword([[0],[1]]))
    print(nn.feedforword([[1],[0]]))
    print(nn.feedforword([[1],[1]]))