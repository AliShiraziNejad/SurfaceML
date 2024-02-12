from surface.simple.sequential import Sequential
from surface.simple.layers import Dense
from surface.simple.activations import ReLU, Softmax
from surface.simple.optimizers import Adam
from surface.simple.losses import CategoricalCrossentropy

from keras.utils import to_categorical
from keras.datasets import mnist

from einops import rearrange

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = rearrange(x_train, 'batch height weight -> batch (height weight)').astype('float32') / 255
    x_test = rearrange(x_test, 'batch height weight -> batch (height weight)').astype('float32') / 255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential()
    model.add(Dense(784, 128, initialization='glorot_normal'))
    model.add(ReLU())
    model.add(Dense(128, 128, initialization='glorot_normal'))
    model.add(ReLU())
    model.add(Dense(128, 10, initialization='glorot_normal'))
    model.add(Softmax())

    optimizer = Adam()
    loss_function = CategoricalCrossentropy()

    # Train the model
    model.train(x_train=x_train, y_train=y_train,
                validation_data=(x_test, y_test),
                epochs=10,
                loss_function=loss_function,
                optimizer=optimizer,
                batch_size=32,
                shuffle=True)
