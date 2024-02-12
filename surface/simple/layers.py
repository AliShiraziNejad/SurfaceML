from surface.simple.initializations import glorot_uniform, glorot_normal, he_uniform, he_normal
import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size, initialization='glorot_uniform'):
        super().__init__()
        if initialization == 'glorot_uniform':
            self.weights = glorot_uniform((input_size, output_size))
        elif initialization == 'glorot_normal':
            self.weights = glorot_normal((input_size, output_size))
        elif initialization == 'he_uniform':
            self.weights = he_uniform((input_size, output_size))
        elif initialization == 'he_normal':
            self.weights = he_normal((input_size, output_size))
        else:
            raise ValueError("Invalid initialization method")
        self.bias = np.zeros(output_size)

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.bias_gradient = np.mean(output_gradient, axis=0)
        input_gradient = np.dot(output_gradient, self.weights.T)
        return input_gradient
