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
        self.weights = [
            Matrix(self.__random_data(n=next_layer_size, m=layer_size))
            for layer_size, next_layer_size in zip(sizes, sizes[1:])
        ]
        self.biases = [
            Matrix([[1]] * layer_size) for layer_size in sizes[1:]
        ]

    @staticmethod
    def __random_data(n, m):
        width = 1 / math.sqrt(n)
        return [
            [random.uniform(-width, width) for i in range(m)]
            for j in range(n)
        ]

    def feedforword(self, input_data):
        if input_data.n != self.sizes[0]:
            raise Exception("Invalid input length")

        output = input_data
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            output = layer_weights * output + layer_biases 
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
        current_layer_output = input_data
        layer_outputs = [input_data]
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            mid_layer_output = (layer_weights * current_layer_output) + layer_biases
            layer_outputs.append(mid_layer_output)
            current_layer_output = mid_layer_output.apply(sigmoid)
            layer_outputs.append(current_layer_output)

        delta_weight, delta_bias  = [], []
        sigmoided_output, layer_output  = layer_outputs.pop(), layer_outputs.pop()
        delta = (sigmoided_output - output) ** layer_output.apply(sigmoid_derivative)
        
        for weight in reversed(self.weights):
            sigmoided_output = layer_outputs.pop()
            delta_bias.insert(0, delta)
            delta_weight.insert(0, delta * sigmoided_output.transpose())
            if layer_outputs:
                layer_output = layer_outputs.pop()
                delta = (weight.transpose() * delta) ** layer_output.apply(sigmoid_derivative)
        return (delta_weight, delta_bias)

    def update_mini_batch(self, batch, learning_rate):
        average_weights = [weight.apply(lambda x: x * 0) for weight in self.weights]
        average_biases = [bais.apply(lambda x: x * 0) for bais in self.biases]

        # summing all the deltas
        for input_data, output in batch:
            delta_weight, delta_bias = self.backpropagation(input_data, output)
            for i in range(len(self.sizes) - 1):
                average_weights[i] += delta_weight[i]
                average_biases[i]  += delta_bias[i]

        # averaging all the deltas
        for i in range(len(self.sizes) - 1):
            average_weights[i] = average_weights[i].apply(lambda x: (learning_rate * x) / len(batch))
            average_biases[i]  = average_biases[i].apply(lambda x: (learning_rate * x) / len(batch))
    
        # subtract the estimated deltas
        for i in range(len(self.sizes) - 1):
            self.weights[i] -= average_weights[i]
            self.biases[i]  -= average_biases[i]
                    
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
            training_output  = self.feedforword(input_data).to_list()
            max_output_index = training_output.index(max(training_output))
            output = output.to_list()
            expected_index   = output.index(max(output))
            print('got output index: {}, expected output index: {}'.format(max_output_index, expected_index))


def Vector(data):
    return Matrix([data]).transpose()


if __name__ == "__main__":
    inputs  = [
        Vector([1, 1]),
        Vector([0, 0]),
        Vector([0, 1]),
        Vector([1, 0]),
    ]
    outputs = [
        Vector([1, 0]),
        Vector([1, 0]),
        Vector([0, 1]),
        Vector([0, 1]),
    ]
    xor_network = NeuralNetwork([inputs[0].n, 4, outputs[0].n])
    
    print("Not traind")
    xor_network.evaluate(inputs, outputs)

    xor_network.train(inputs, outputs, learning_rate=0.8, epochs=5000, mini_batch_size=4)
    
    print("Traind")
    xor_network.evaluate(inputs, outputs)
