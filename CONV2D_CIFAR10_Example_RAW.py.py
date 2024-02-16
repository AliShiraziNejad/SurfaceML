import numpy as np

from keras.utils import to_categorical
from keras.datasets import cifar10, mnist

from einops import rearrange

from scipy.stats import truncnorm

import tensorflow as tf
import random
import os

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def train(self, x_train, y_train, epochs, loss_function, optimizer, batch_size=32, validation_data=None, shuffle=True):
        for epoch in range(epochs):
            if shuffle:
                permutation = np.random.permutation(x_train.shape[0])
                x_train_set = x_train[permutation]
                y_train_set = y_train[permutation]
            else:
                x_train_set = x_train
                y_train_set = y_train

            total_loss = 0
            correct_predictions = 0

            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train_set[i:i + batch_size]
                y_batch = y_train_set[i:i + batch_size]

                output = self.forward(x_batch)
                loss = loss_function.loss(y_batch, output)
                total_loss += loss

                predictions = np.argmax(output, axis=1)
                labels = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(predictions == labels)

                output_gradient = loss_function.gradient(y_batch, output)
                self.backward(output_gradient)

                for layer_id, layer in enumerate(self.layers):
                    if hasattr(layer, 'weights'):
                        optimizer.update(layer_id, layer.weights, layer.bias, layer.weights_gradient,
                                         layer.bias_gradient)

            epoch_loss = total_loss / (x_train.shape[0] / batch_size)
            epoch_accuracy = correct_predictions / x_train.shape[0]

            if validation_data is not None:
                test_output = self.forward(validation_data[0])
                test_loss = loss_function.loss(validation_data[1], test_output)
                test_predictions = np.argmax(test_output, axis=1)
                test_labels = np.argmax(validation_data[1], axis=1)
                test_accuracy = np.mean(test_predictions == test_labels)

                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError


def calculate_fan_in_fan_out(shape):
    if len(shape) == 2:  # Linear layer
        fan_in, fan_out = shape[1], shape[0]
    else:  # Conv1D, Conv2D, Conv3D
        receptive_field_size = np.prod(shape[2:])  # Product of kernel dimensions
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    return fan_in, fan_out


def glorot_uniform(shape):
    fan_in, fan_out = calculate_fan_in_fan_out(shape)
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


def glorot_normal(shape):
    fan_in, fan_out = calculate_fan_in_fan_out(shape)
    stddev = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, stddev, size=shape)


def he_uniform(shape):
    fan_in, _ = calculate_fan_in_fan_out(shape)
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


def he_normal(shape):
    fan_in, _ = calculate_fan_in_fan_out(shape)
    stddev = np.sqrt(2 / fan_in)
    return np.random.normal(0, stddev, size=shape)


def trunc_norm(shape, mean=0.0, stddev=None, lower=-2.0, upper=2.0):
    fan_in, _ = calculate_fan_in_fan_out(shape)
    if stddev is None:
        stddev = np.sqrt(1. / fan_in)  # He initialization strategy
    lower_bound = (lower - mean) / stddev
    upper_bound = (upper - mean) / stddev
    distribution = truncnorm(a=lower_bound, b=upper_bound, loc=mean, scale=stddev)
    return distribution.rvs(size=shape)


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_gradient):
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


class Flatten(Layer):
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)


class GlobalAveragePooling2D(Layer):
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return np.mean(input_data, axis=(2, 3), keepdims=False)

    def backward(self, output_gradient):
        n, c, h, w = self.input_shape
        return np.repeat(output_gradient[:, :, np.newaxis, np.newaxis], h * w).reshape(self.input_shape) / (h * w)


class Adam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_weights = {}
        self.v_weights = {}
        self.m_biases = {}
        self.v_biases = {}

    def update(self, layer_id, weights, biases, weights_gradient, biases_gradient):
        if layer_id not in self.m_weights:
            self.m_weights[layer_id] = np.zeros_like(weights)
            self.v_weights[layer_id] = np.zeros_like(weights)
            self.m_biases[layer_id] = np.zeros_like(biases)
            self.v_biases[layer_id] = np.zeros_like(biases)

        self.m_weights[layer_id] = self.beta_1 * self.m_weights[layer_id] + (1 - self.beta_1) * weights_gradient
        self.v_weights[layer_id] = self.beta_2 * self.v_weights[layer_id] + (1 - self.beta_2) * (weights_gradient ** 2)

        self.m_biases[layer_id] = self.beta_1 * self.m_biases[layer_id] + (1 - self.beta_1) * biases_gradient
        self.v_biases[layer_id] = self.beta_2 * self.v_biases[layer_id] + (1 - self.beta_2) * (biases_gradient ** 2)

        m_hat_weights = self.m_weights[layer_id] / (1 - self.beta_1)
        v_hat_weights = self.v_weights[layer_id] / (1 - self.beta_2)

        m_hat_biases = self.m_biases[layer_id] / (1 - self.beta_1)
        v_hat_biases = self.v_biases[layer_id] / (1 - self.beta_2)

        weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

    def reset(self):
        """
        Not used for sequential class training!
        """
        self.m_weights.clear()
        self.v_weights.clear()
        self.m_biases.clear()
        self.v_biases.clear()


class CategoricalCrossentropy:
    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def loss(self, y_true, y_pred):
        if self.from_logits:
            y_pred = self.softmax(y_pred)
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

    def gradient(self, y_true, y_pred):
        if self.from_logits:
            y_pred = self.softmax(y_pred)
        return y_pred - y_true

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train[:100].astype('float32') / 255
    y_train = y_train[:100]
    x_test = x_test[:100].astype('float32') / 255
    y_test = y_test[:100]

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    x_train = rearrange(x_train, 'batch height width channels -> batch channels height width')
    x_test = rearrange(x_test, 'batch height width channels -> batch channels height width')

    model = Sequential()
    model.add(Conv2D(input_channels=3, output_channels=32, kernel_size=3, stride=1, padding=1, initialization='he_normal'))
    model.add(ReLU())
    model.add(Conv2D(input_channels=32, output_channels=64, kernel_size=3, stride=1, padding=1, initialization='he_normal'))
    model.add(ReLU())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, 128, initialization='he_normal'))
    model.add(ReLU())
    model.add(Dense(128, 10, initialization='he_normal'))
    model.add(Softmax())

    optimizer = Adam()
    loss_function = CategoricalCrossentropy()

    model.train(x_train=x_train, y_train=y_train,
                validation_data=(x_test, y_test),
                epochs=50,
                loss_function=loss_function,
                optimizer=optimizer,
                batch_size=256,
                shuffle=True)
