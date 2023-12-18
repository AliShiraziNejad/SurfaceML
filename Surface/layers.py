from initializations import glorot_uniform, glorot_normal, he_uniform, he_normal
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


class Conv1D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, padding='valid', initialization='glorot_uniform'):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        shape = (kernel_size, input_channels, output_channels)

        if initialization == 'glorot_uniform':
            self.weights = glorot_uniform(shape)
        elif initialization == 'glorot_normal':
            self.weights = glorot_normal(shape)
        elif initialization == 'he_uniform':
            self.weights = he_uniform(shape)
        elif initialization == 'he_normal':
            self.weights = he_normal(shape)
        else:
            raise ValueError("Invalid initialization method")

        self.bias = np.zeros(output_channels)

    def forward(self, input_data):
        if self.padding == 'same':
            padding_width = (self.kernel_size - 1) // 2
            padded_input = np.pad(input_data, ((padding_width, padding_width), (0, 0)), 'constant')
        else:
            padded_input = input_data

        self.input = padded_input
        output_length = input_data.shape[0] - self.kernel_size + 1
        self.output = np.zeros((output_length, self.output_channels))

        for i in range(output_length):
            for j in range(self.output_channels):
                self.output[i, j] = np.sum(padded_input[i:i + self.kernel_size, :] * self.weights[:, :, j]) + self.bias[
                    j]

        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros_like(self.input)
        weights_gradient = np.zeros_like(self.weights)
        bias_gradient = np.zeros_like(self.bias)

        output_length = input_gradient.shape[0] - self.kernel_size + 1

        for i in range(output_length):
            for j in range(self.output_channels):
                weights_gradient[:, :, j] += self.input[i:i + self.kernel_size, :] * output_gradient[i, j]
                input_gradient[i:i + self.kernel_size, :] += self.weights[:, :, j] * output_gradient[i, j]
                bias_gradient[j] += output_gradient[i, j]

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient
