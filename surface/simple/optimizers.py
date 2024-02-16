import numpy as np


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


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_weights = {}
        self.velocity_biases = {}

    def update(self, layer_id, weights, biases, weights_gradient, biases_gradient):
        if self.momentum:
            if layer_id not in self.velocity_weights:
                self.velocity_weights[layer_id] = np.zeros_like(weights)
                self.velocity_biases[layer_id] = np.zeros_like(biases)

            self.velocity_weights[layer_id] = self.momentum * self.velocity_weights[
                layer_id] - self.learning_rate * weights_gradient
            self.velocity_biases[layer_id] = self.momentum * self.velocity_biases[
                layer_id] - self.learning_rate * biases_gradient

            weights += self.velocity_weights[layer_id]
            biases += self.velocity_biases[layer_id]
        else:
            weights -= self.learning_rate * weights_gradient
            biases -= self.learning_rate * biases_gradient

    def reset(self):
        self.velocity_weights.clear()
        self.velocity_biases.clear()


class RMSProp:
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.sq_grad_weights = {}
        self.sq_grad_biases = {}

    def update(self, layer_id, weights, biases, weights_gradient, biases_gradient):
        if layer_id not in self.sq_grad_weights:
            self.sq_grad_weights[layer_id] = np.zeros_like(weights)
            self.sq_grad_biases[layer_id] = np.zeros_like(biases)

        self.sq_grad_weights[layer_id] = self.rho * self.sq_grad_weights[layer_id] + (1 - self.rho) * (
                weights_gradient ** 2)
        self.sq_grad_biases[layer_id] = self.rho * self.sq_grad_biases[layer_id] + (1 - self.rho) * (
                biases_gradient ** 2)

        weights -= self.learning_rate * weights_gradient / (np.sqrt(self.sq_grad_weights[layer_id]) + self.epsilon)
        biases -= self.learning_rate * biases_gradient / (np.sqrt(self.sq_grad_biases[layer_id]) + self.epsilon)

    def reset(self):
        self.sq_grad_weights.clear()
        self.sq_grad_biases.clear()
