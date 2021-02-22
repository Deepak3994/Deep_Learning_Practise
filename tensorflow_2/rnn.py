from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, SimpleRNN, TimeDistributed
from tensorflow.keras.models import Model, Sequential
import numpy as np

# input shape
inputs = Input(shape=(1,2))
# output shape, return state, use tanh as activation function
output, state = SimpleRNN(3, return_state=True, activation='tanh')(inputs)
model = Model(inputs=inputs, outputs=[output, state])
# test input
data = np.array([[ [1,2] ]])
# print output, state
output, state = model.predict(data)
print("output: ",output)
print("state: ",state)

# weights for input
model.layers[1].weights[0]

# weights for state
model.layers[1].weights[1]

# bias
model.layers[1].weights[2]


# Sequence tagging example
John = [1,0,0]
loves = [0,1,0]
Jane = [0,0,1]

X = np.array([
    [ John, loves, Jane ],
    [ Jane, loves, John ]
]).astype(np.float32)

S = [0] # subject
V = [1] # verb
O = [2] # object
y = np.array([[S, V, O], [S, V, O]]).astype(np.float32)

# input shape
inputs = Input(shape=(3, 3))
# output shape, return state, return sequence
output, state = SimpleRNN(3, return_state=True, return_sequences=True)(inputs)
model = Model(inputs=inputs, outputs=[output, state])

# print output, state
output, state = model.predict(X)

print("John loves Jane: ",output[0])
print("Jane loves John: ",output[1])

# the state value is same with the last output
print("John loves Jane: state: ",state[0])
print("Jane loves John: state: ",state[1])

model = Sequential()
model.add(SimpleRNN(3, input_shape=(3, 3), return_sequences=True))
model.add(TimeDistributed(Dense(3, activation="softmax")))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
print(model.summary())
# train, takes 30sec, if you want to monitor progreses, change verbose=1
model.fit(X, y, epochs=2000, verbose=0)

result = model.predict(X, verbose=0)


# check the result

# 0 : Subject
# 1 : Verb
# 2 : Object
np.argmax(result, axis=1)

# Sentence classifciation
from IPython.display import Image

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

tf.random.set_seed(1)
np.random.seed(1)

movie_reviews = [
         {'review': 'this is the best movie', 'sentiment': 'positive'},
         {'review': 'i recommend you watch this movie', 'sentiment': 'positive'},
         {'review': 'it was waste of money and time', 'sentiment': 'negative'},
         {'review': 'the worst movie ever', 'sentiment': 'negative'}
    ]
df = pd.DataFrame(movie_reviews)

def get_vocab2int(df):
    d = {}
    vocab = set()
    df['review'].str.split().apply(vocab.update)
    for idx, word in enumerate(vocab):
        d[word] = idx
    return d

vocab2_int = get_vocab2int(df)
vocab_size = len(vocab2_int)
# encode words into integer
reviews = df['review'].tolist()
encoded_reviews = []
for review in reviews:
    tokens = review.split(" ")
    review_encoding = []
    for token in tokens:
        review_encoding.append(vocab2_int[token])
    encoded_reviews.append(review_encoding)

# encoded reviews
print(encoded_reviews[0])
print(encoded_reviews[1])
print(encoded_reviews[2])
print(encoded_reviews[3])

def get_max_length(df):
    max_length = 0
    for row in df['review']:
        if len(row.split(" ")) > max_length:
            max_length = len(row.split(" "))
    return max_length

# max_length is used for max sequence of input
max_length = get_max_length(df)
# if review is short, fill in zero padding and make all sentence length to be same as max_length
padded_reviews_encoding = pad_sequences(encoded_reviews, maxlen=max_length, padding='post')
sentiments = df['sentiment'].tolist()
def sentiment_encode(sentiment):
    if sentiment == 'positive':
        return [1,0]
    else:
        return [0,1]


# encoded sentiment
encoded_sentiment = [sentiment_encode(sentiment) for sentiment in sentiments]
# RNN model
model = Sequential()
model.add(Embedding(vocab_size, 3, input_length=max_length))
model.add(SimpleRNN(32))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
train_X = np.array(padded_reviews_encoding)
train_Y = np.array(encoded_sentiment)
print('Train...')
model.fit(train_X, train_Y,epochs=50)
score, acc = model.evaluate(train_X, train_Y, verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)

