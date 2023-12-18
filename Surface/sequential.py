import numpy as np


class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def train(self, x_train, y_train, epochs, loss_function, optimizer, batch_size=32, x_test=None, y_test=None):
        for epoch in range(epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            total_loss = 0
            correct_predictions = 0

            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                output = self.forward(x_batch)
                loss = loss_function.loss(y_batch, output)
                total_loss += loss

                predictions = np.argmax(output, axis=1)
                labels = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(predictions == labels)

                output_gradient = loss_function.gradient(y_batch, output)
                self.backward(output_gradient, optimizer.learning_rate)

                for layer_id, layer in enumerate(self.layers):
                    if hasattr(layer, 'weights'):
                        optimizer.update(layer_id, layer.weights, layer.bias, layer.weights_gradient,
                                         layer.bias_gradient)

            epoch_loss = total_loss / (x_train.shape[0] / batch_size)
            epoch_accuracy = correct_predictions / x_train.shape[0]

            if x_test is not None and y_test is not None:
                test_output = self.forward(x_test)
                test_loss = loss_function.loss(y_test, test_output)
                test_predictions = np.argmax(test_output, axis=1)
                test_labels = np.argmax(y_test, axis=1)
                test_accuracy = np.mean(test_predictions == test_labels)

                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
