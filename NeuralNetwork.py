from Matrix import Matrix
import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = []
        self.biases = []
        for i in range(1, len(sizes)):
            self.weights.append(Matrix(
                self.__random_data(n=sizes[i], m=sizes[i-1])
            ))
            # All ones
            self.biases.append(Matrix(
                [[1]] * sizes[i]
            ))

    @staticmethod
    def __random_data(n, m):
        width = 1 / math.sqrt(n)
        return [
            [random.uniform(-width, width) for i in range(m)]
            for j in range(n)
        ]

    def feedforword(self, input_data):
        if len(input_data) != self.sizes[0]:
            raise Exception("Invalid input length")

        input_data = Matrix([input_data])

        output = input_data.transpose()
        for i in range(len(self.sizes) - 1):
            output = (self.weights[i] * output) + self.biases[i]
            output = output.apply(sigmoid)
        return output
        
    def backpropagation(self, input_data, output):
        """
        delta_new:

                weight.transpose()
                *
                delta
            **
                output.apply(sigmoid_derivative)
        *
            sigmoided_output.transpose()
        """
        layer_output = input_data
        layers = [input_data]
        for weight, bias in zip(self.weights,self.biases):
            layer_output = (weight * layer_output) + bias
            layers.append(layer_output)
            layer_output = layer_output.apply(sigmoid)
            layers.append(layer_output)

        delta_weight, delta_bias  = [], []
        sigmoided_output, layer_output  = layers.pop(), layers.pop()
        delta = (sigmoided_output - output) ** layer_output.apply(sigmoid_derivative)
        
        for weight in reversed(self.weights):
            sigmoided_output = layers.pop()
            delta_bias.insert(0, delta)
            delta_weight.insert(0, delta * sigmoided_output.transpose())
            if layers:
                layer_output = layers.pop()
                delta = (weight.transpose() * delta) ** layer_output.apply(sigmoid_derivative)
        return (delta_weight, delta_bias)

    def update_mini_batch(self, batch, learning_rate):
        average_weights = [weight.apply(lambda x: 0) for weight in self.weights]
        average_bias = [bais.apply(lambda x: 0) for bais in self.biases]

        # summing all the deltas
        for input_data, output in batch:
            input_data = Matrix([input_data]).transpose()
            output = Matrix([output]).transpose()
            delta_weight, delta_bias = self.backpropagation(input_data, output)
            for i in range(len(self.sizes) - 1):
                average_weights[i] += delta_weight[i]
                average_bias[i]    += delta_bias[i]

        # averaging all the deltas
        for i in range(len(self.sizes) - 1):
            average_weights[i] = average_weights[i].apply(lambda x: (learning_rate * x) / len(batch))
            average_bias[i]    = average_bias[i].apply(lambda x: (learning_rate * x) / len(batch))
    
        # subtract the estimated deltas
        for i, weight, bais in zip(range(len(self.sizes)), average_weights, average_bias):
            self.weights[i] -= weight
            self.biases[i]  -= bais
                    
    def train(self, inputs, outputs, learning_rate=0.5, epochs=100, mini_batch_size=None, verbose=False):
        if not mini_batch_size:
            mini_batch_size = len(inputs)

        data = list(zip(inputs, outputs))
        for i in range(epochs):
            random.shuffle(data)
            mini_batches = [
                data[batch_start_index: batch_start_index + mini_batch_size]
                for batch_start_index in range(0, len(data), mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if verbose:
                print("Epoch:", i + 1)


    def evaluate(self, inputs, outputs):
        for input_data, output in zip(inputs, outputs):
            training_output  = self.feedforword(input_data)
            max_output_index = training_output.data.index(max(training_output.data))
            expected_index   = output.index(max(output))
            print('got output index: {}, expected output index: {}'.format(max_output_index, expected_index))

if __name__ == "__main__":
    inputs  = [
        [1,1],
        [0,0],
        [0,1],
        [1,0]
    ]
    outputs = [
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ]
    xor_network = NeuralNetwork([len(inputs[0]), 4, len(outputs[0])])
    
    print("Not traind")
    xor_network.evaluate(inputs, outputs)

    xor_network.train(inputs, outputs, learning_rate=0.8, epochs=5000, mini_batch_size=4)
    
    print("Traind")
    xor_network.evaluate(inputs, outputs)
