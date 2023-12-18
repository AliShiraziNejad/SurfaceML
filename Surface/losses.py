import numpy as np


class MeanSquaredError:
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    @staticmethod
    def gradient(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class MeanAbsoluteError:
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def gradient(y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.size


class BinaryCrossentropy:
    @staticmethod
    def loss(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def gradient(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))


class CategoricalCrossentropy:
    @staticmethod
    def loss(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

    @staticmethod
    def gradient(y_true, y_pred):
        return y_pred - y_true


class SparseCategoricalCrossentropy:
    @staticmethod
    def loss(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.log(y_pred[np.arange(y_true.shape[0]), y_true.astype(int)]))

    @staticmethod
    def gradient(y_true, y_pred):
        y_pred[np.arange(y_true.shape[0]), y_true.astype(int)] -= 1
        return y_pred


class HingeLoss:
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean(np.maximum(1 - y_true * y_pred, 0))

    @staticmethod
    def gradient(y_true, y_pred):
        gradient = np.zeros_like(y_pred)
        gradient[y_true * y_pred < 1] = -y_true[y_true * y_pred < 1]
        return gradient


class HuberLoss:
    def __init__(self, delta=1.0):
        self.delta = delta

    @staticmethod
    def loss(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = np.abs(error) < self.delta
        squared_loss = 0.5 * np.square(error)
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        return np.where(is_small_error, squared_loss, linear_loss)

    @staticmethod
    def gradient(self, y_true, y_pred):
        error = y_true - y_pred
        return np.where(np.abs(error) < self.delta, error, np.sign(error) * self.delta)


class LogCosh:
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean(np.log(np.cosh(y_pred - y_true)))

    @staticmethod
    def gradient(y_true, y_pred):
        return np.tanh(y_pred - y_true)
