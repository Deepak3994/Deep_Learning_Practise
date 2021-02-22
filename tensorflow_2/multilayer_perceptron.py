import numpy as np
import tensorflow as tf
tf.random.set_seed(1)
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.models import Model, Sequential

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
scores = [1, 7, 2, 1, 1, 1, 1, 3, 1, 1]
print(softmax(scores))

# get MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale

# MLP model for classification
model = Sequential([Flatten(input_shape=(28, 28)),
                    Dense(256, activation="sigmoid"),
                    Dense(128, activation="sigmoid"),
                    Dense(10, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# test
results = model.evaluate(x_test, y_test, verbose=0)
print("test_loss, test-acc : ", results)
