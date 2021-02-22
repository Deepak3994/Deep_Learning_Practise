from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# random seed for always same results
tf.random.set_seed(678)
import numpy as np

# Inputs and Labels
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = np.array([0.,1.,1.,0.])


# Two dense layers
model = Sequential()
# first dense layer
model.add(Dense(units=2, activation="sigmoid", input_dim=2))
# second dense layer
model.add(Dense(units=1, activation="sigmoid"))
# loss function and optimization
model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
print(model.summary())

# train
model.fit(X, y, epochs=50000, batch_size=4, verbose=0)
print(model.predict(X, batch_size=4))

# printing first dense layer weights and bias
print("first layer weights: ",model.layers[0].get_weights()[0])
print("first layer bias: ",model.layers[0].get_weights()[1])

# printing second dense layer weights and bias
print("first layer weights: ",model.layers[1].get_weights()[0])
print("first layer bias: ",model.layers[1].get_weights()[1])

