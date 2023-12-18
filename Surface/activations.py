from surface.layers import Layer
import numpy as np


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return self.activation_prime(self.input) * output_gradient


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0) * 1

        super().__init__(relu, relu_prime)


class Softmax(Activation):
    def __init__(self):
        def softmax(x):
            exps = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)

        def softmax_prime(x):
            return np.ones_like(x)

        super().__init__(softmax, softmax_prime)

    def backward(self, output_gradient, learning_rate):
        return output_gradient


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1.0 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Linear(Activation):
    def __init__(self):
        def linear(x):
            return x

        def linear_prime(x):
            return np.ones_like(x)

        super().__init__(linear, linear_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
