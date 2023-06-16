from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import re
import pandas as pd
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords

import re
from keras.models import load_model
import numpy as np

stop_words = set(stopwords.words('indonesian'))

def cleansing(sent):
  string = sent.lower()
  string = re.sub(r'[^a-zA-z0-9]',' ', string)
  string = re.sub(r'[^\w]',' ', string)

  words = nltk.word_tokenize(string)
  words = [word for word in words if word not in stop_words]

  text = ' '.join(words)
  return text

#MLP

import pickle
with open("models/mlp/feature.p", "rb") as file:
  count_vect = pickle.load(file)

with open("models/mlp/model.p", "rb") as file:
  model_mlp = pickle.load(file)


async def get_sentiment(input, type):
    if type == 'mlp':
        original_text = input
        text = count_vect.transform([cleansing(original_text)])
        result = model_mlp.predict(text)[0]
        return result
    else:
        try:
            if type == 'rnn':
                model = load_model('models/rnn/model_rnn.h5')
            else:
                model = load_model('models/lstm/model_lstm.h5')

            max_features = 100000
            tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

            file = open('models/lstm/x_pad_sequences.pickle', 'rb')
            X = pickle.load(file)

            input_text = input
            sentiment = ['negative', 'neutral', 'positive']

            text = [cleansing(input_text)]
            predicted = tokenizer.texts_to_sequences(text)
            guess = pad_sequences(predicted, maxlen=X.shape[1])

            prediction = model.predict(guess)
            polarity = np.argmax(prediction[0])

            return sentiment[polarity]
        except Exception as e:
            print(e)


async def get_sentiment_file(input, type):
    if type == 'mlp':
        original_text = input.loc[0, 'text']
        text = count_vect.transform([cleansing(original_text)])
        result = model_mlp.predict(text)[0]
        return original_text, result
    else:
        try:
            if type == 'rnn':
                model = load_model('models/rnn/model_rnn.h5')
            else:
                model = load_model('models/lstm/model_lstm.h5')

            max_features = 100000
            tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

            file = open('models/lstm/x_pad_sequences.pickle', 'rb')
            X = pickle.load(file)

            input_text = input.loc[0, 'text']
            sentiment = ['negative', 'neutral', 'positive']

            text = [cleansing(input_text)]
            predicted = tokenizer.texts_to_sequences(text)
            guess = pad_sequences(predicted, maxlen=X.shape[1])

            prediction = model.predict(guess)
            polarity = np.argmax(prediction[0])

            return input_text, sentiment[polarity]
        except Exception as e:
            print(e)


