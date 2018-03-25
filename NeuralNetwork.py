from Matrix import Matrix
import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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

if __name__ == "__main__":
    nn = NeuralNetwork([2,3,1])
    print(nn.feedforword([[0],[0]]))