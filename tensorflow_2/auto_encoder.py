from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import matplotlib .pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# inorder to get the same result
tf.random.set_seed(1)
np.random.seed(1)


# get MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# MNIST preprocess
x_train =  x_train.reshape(60000, 784)

# select only 300 test data for visualization
x_test = x_test[:300]
y_test = y_test[:300]
x_test = x_test.reshape(300, 784)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# normalise the data
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale

# modeling
# MNIST input 28 rows and 28 columns = 784 pixels
input_img = Input(shape=(784,))
# encoder
encoder1 = Dense(128, activation="sigmoid")(input_img)
encoder2 = Dense(3, activation="sigmoid")(encoder1)
# decoer
decoder1 = Dense(128, activation="sigmoid")(encoder2)
decoder2 = Dense(784, activation="sigmoid")(decoder1)

#this model maps an input to its reconstruction
autoencoder = Model(inputs=input_img, outputs=decoder2)
autoencoder.compile(optimizer="adam",  loss="binary_crossentropy")
autoencoder.fit(x_train, x_train, epochs=5, batch_size=32,
                shuffle=True, validation_data=(x_test, x_test))


# create encoder model
encoder = Model(inputs=input_img, outputs=encoder2)
# create decoder model
encoder_input = Input(shape=(3,))
decoder_layer1 = autoencoder.layers[-2]
decoder_layer2 = autoencoder.layers[-1]
decoder = Model(inputs=encoder_input, outputs=decoder_layer2(decoder_layer1(encoder_input)))

# get latent vector for visualization
latent_vector = encoder.predict(x_test)
# get decoder output to visualize reconstructed image
reconstructed_imgs = decoder.predict(latent_vector)

# visualize in 3D plot
from pylab import rcParams

rcParams['figure.figsize'] = 10, 8

fig = plt.figure(1)
ax = Axes3D(fig)

xs = latent_vector[:, 0]
ys = latent_vector[:, 1]
zs = latent_vector[:, 2]

color = ['red', 'green', 'blue', 'lime', 'white', 'pink', 'aqua', 'violet', 'gold', 'coral']

for x, y, z, label in zip(xs, ys, zs, y_test):
    c = color[int(label)]
    ax.text(x, y, z, label, backgroundcolor=c)

ax.set_xlim(xs.min(), xs.max())
ax.set_ylim(ys.min(), ys.max())
ax.set_zlim(zs.min(), zs.max())

plt.show()

# visualize re-constructed image
# visualize in 3D plot
from pylab import rcParams

rcParams['figure.figsize'] = 10, 8

fig = plt.figure(1)
ax = Axes3D(fig)

xs = latent_vector[:, 0]
ys = latent_vector[:, 1]
zs = latent_vector[:, 2]

color = ['red', 'green', 'blue', 'lime', 'white', 'pink', 'aqua', 'violet', 'gold', 'coral']

for x, y, z, label in zip(xs, ys, zs, y_test):
    c = color[int(label)]
    ax.text(x, y, z, label, backgroundcolor=c)

ax.set_xlim(xs.min(), xs.max())
ax.set_ylim(ys.min(), ys.max())
ax.set_zlim(zs.min(), zs.max())

plt.show()
