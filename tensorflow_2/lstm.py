from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import numpy as np
# Load Pretrained Word2Vec
embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")


def get_max_length(df):
    """
    get max token counts from train data,
    so we use this number as fixed length input to RNN cell
    """
    max_length = 0
    for row in df['review']:
        if len(row.split(" ")) > max_length:
            max_length = len(row.split(" "))
    return max_length


def get_word2vec_enc(reviews):
    """
    get word2vec value for each word in sentence.
    concatenate word in numpy array, so we can use it as RNN input
    """
    encoded_reviews = []
    for review in reviews:
        tokens = review.split(" ")
        word2vec_embedding = embed(tokens)
        encoded_reviews.append(word2vec_embedding)
    return encoded_reviews


def get_padded_encoded_reviews(encoded_reviews):
    """
    for short sentences, we prepend zero padding so all input to RNN has same length
    """
    padded_reviews_encoding = []
    for enc_review in encoded_reviews:
        zero_padding_cnt = max_length - enc_review.shape[0]
        pad = np.zeros((1, 250))
        for i in range(zero_padding_cnt):
            enc_review = np.concatenate((pad, enc_review), axis=0)
        padded_reviews_encoding.append(enc_review)
    return padded_reviews_encoding


def sentiment_encode(sentiment):
    """
    return one hot encoding for Y value
    """
    if sentiment == 'positive':
        return [1, 0]
    else:
        return [0, 1]


def preprocess(df):
    """
    encode text value to numeric value
    """
    # encode words into word2vec
    reviews = df['review'].tolist()

    encoded_reviews = get_word2vec_enc(reviews)
    padded_encoded_reviews = get_padded_encoded_reviews(encoded_reviews)
    # encoded sentiment
    sentiments = df['sentiment'].tolist()
    encoded_sentiment = [sentiment_encode(sentiment) for sentiment in sentiments]
    X = np.array(padded_encoded_reviews)
    Y = np.array(encoded_sentiment)
    return X, Y


# preprocesss text to number
movie_reviews_train = [
    {'review': 'this is the best movie', 'sentiment': 'positive'},
    {'review': 'i recommend you watch this movie', 'sentiment': 'positive'},
    {'review': 'it was waste of money and time', 'sentiment': 'negative'},
    {'review': 'the worst movie ever', 'sentiment': 'negative'}
]
df = pd.DataFrame(movie_reviews_train)

# max_length is used for max sequence of input
max_length = get_max_length(df)

train_X, train_Y = preprocess(df)

# LSTM model
model = Sequential()
model.add(LSTM(32))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(train_X, train_Y,epochs=50)
model.summary()

"""
movie_reviews_train = [
         {'review': 'this is the best movie', 'sentiment': 'positive'},
         {'review': 'i recommend you watch this movie', 'sentiment': 'positive'},
         {'review': 'it was waste of money and time', 'sentiment': 'negative'},
         {'review': 'the worst movie ever', 'sentiment': 'negative'}
    ]
"""
movie_reviews_test = [
         {'review': 'it is better movie', 'sentiment': 'positive'},
         {'review': 'i suggest you see this movie', 'sentiment': 'positive'},
         {'review': 'it was just throwing 20 dollars away', 'sentiment': 'negative'},
         {'review': 'worse than any show', 'sentiment': 'negative'},
         {'review': 'nice movie, so love it', 'sentiment': 'positive'},
         {'review': 'terrible', 'sentiment': 'negative'}
    ]
test_df = pd.DataFrame(movie_reviews_test)

test_X, test_Y = preprocess(test_df)

score, acc = model.evaluate(test_X, test_Y, verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)
