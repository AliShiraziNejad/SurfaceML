from surface.simple.initializations import glorot_uniform, glorot_normal, he_uniform, he_normal, trunc_norm
import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient):
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
        elif initialization == 'trunc_norm':
            self.weights = trunc_norm((input_size, output_size))
        else:
            raise ValueError("Invalid initialization method")
        self.bias = np.zeros(output_size)

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient):
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.bias_gradient = np.mean(output_gradient, axis=0)
        input_gradient = np.dot(output_gradient, self.weights.T)
        return input_gradient


class Flatten(Layer):
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)


class Conv1D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, initialization='glorot_uniform'):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if initialization == 'glorot_uniform':
            self.weights = glorot_uniform((output_channels, input_channels, kernel_size))
        elif initialization == 'glorot_normal':
            self.weights = glorot_normal((output_channels, input_channels, kernel_size))
        elif initialization == 'he_uniform':
            self.weights = he_uniform((output_channels, input_channels, kernel_size))
        elif initialization == 'he_normal':
            self.weights = he_normal((output_channels, input_channels, kernel_size))
        elif initialization == 'trunc_norm':
            self.weights = trunc_norm((output_channels, input_channels, kernel_size))
        else:
            raise ValueError("Invalid initialization method")
        self.bias = np.zeros(output_channels)

    def forward(self, input_data):
        self.input = input_data
        batch_size, channels, width = input_data.shape
        out_width = (width - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, self.output_channels, out_width))

        for i in range(out_width):
            start_i = i * self.stride
            end_i = start_i + self.kernel_size
            output[:, :, i] = np.tensordot(input_data[:, :, start_i:end_i], self.weights, axes=([1, 2], [1, 2])) + self.bias

        self.output = output
        return output

    def backward(self, output_gradient):
        batch_size, channels, width = self.input.shape
        _, _, out_width = output_gradient.shape
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
        input_gradient = np.zeros_like(self.input)

        for i in range(out_width):
            start_i = i * self.stride
            end_i = start_i + self.kernel_size
            self.weights_gradient += np.tensordot(output_gradient[:, :, i], self.input[:, :, start_i:end_i], axes=([0], [0]))
            self.bias_gradient += output_gradient[:, :, i].sum(axis=0)
            input_gradient[:, :, start_i:end_i] += np.tensordot(output_gradient[:, :, i], self.weights, axes=([1], [0]))

        return input_gradient


class GlobalAveragePooling1D(Layer):
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return np.mean(input_data, axis=2, keepdims=False)

    def backward(self, output_gradient):
        n = self.input_shape[2]
        input_gradient = np.repeat(output_gradient[:, :, np.newaxis], n, axis=2) / n
        return input_gradient


class Conv2D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, initialization='glorot_uniform'):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initialization = initialization

        kernel_shape = (output_channels, input_channels, kernel_size, kernel_size)

        if initialization == 'glorot_uniform':
            self.weights = glorot_uniform(kernel_shape)
        elif initialization == 'glorot_normal':
            self.weights = glorot_normal(kernel_shape)
        elif initialization == 'he_uniform':
            self.weights = he_uniform(kernel_shape)
        elif initialization == 'he_normal':
            self.weights = he_normal(kernel_shape)
        elif initialization == 'trunc_norm':
            self.weights = trunc_norm(kernel_shape)
        else:
            raise ValueError("Invalid initialization method")
        self.bias = np.zeros(output_channels)

    def forward(self, input_data):
        self.input = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        batch_size, _, input_height, input_width = self.input.shape
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1

        self.output = np.zeros((batch_size, self.output_channels, output_height, output_width))

        for i in range(batch_size):
            for j in range(self.output_channels):
                for k in range(0, input_height - self.kernel_size + 1, self.stride):
                    for l in range(0, input_width - self.kernel_size + 1, self.stride):
                        self.output[i, j, k // self.stride, l // self.stride] = np.sum(
                            self.input[i, :, k:k + self.kernel_size, l:l + self.kernel_size] * self.weights[j, :, :, :]
                        ) + self.bias[j]

        return self.output

    def backward(self, output_gradient):
        batch_size, _, input_height, input_width = self.input.shape
        _, _, output_height, output_width = output_gradient.shape

        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
        input_gradient = np.zeros_like(self.input)

        for i in range(batch_size):
            for j in range(self.output_channels):
                for k in range(output_height):
                    for l in range(output_width):
                        input_slice = self.input[i, :, k * self.stride:k * self.stride + self.kernel_size, l * self.stride:l * self.stride + self.kernel_size]
                        self.weights_gradient[j, :, :, :] += output_gradient[i, j, k, l] * input_slice
                        input_gradient[i, :, k * self.stride:k * self.stride + self.kernel_size, l * self.stride:l * self.stride + self.kernel_size] += output_gradient[i, j, k, l] * self.weights[j, :,
                                                                                                                                                                                      :, :]
                self.bias_gradient[j] += np.sum(output_gradient[i, j, :, :])

        if self.padding > 0:
            input_gradient = input_gradient[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return input_gradient


class GlobalAveragePooling2D(Layer):
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return np.mean(input_data, axis=(2, 3), keepdims=False)

    def backward(self, output_gradient):
        n, c, h, w = self.input_shape
        return np.repeat(output_gradient[:, :, np.newaxis, np.newaxis], h * w).reshape(self.input_shape) / (h * w)
