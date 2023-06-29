import re 
import pickle 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pathlib import Path

import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))

#load Data
df= pd.read_csv('data/train_preprocess.tsv.txt', sep='\t', header=None)
df.columns =['text', 'label']

#Cleansing
def lowercase(text):
    return text.lower()
def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('user',' ',text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    return text
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text
def preprocess(text):
    text = lowercase(text) 
    text = remove_nonaplhanumeric(text) 
    text = remove_unnecessary_char(text) 
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text
df['text_clean'] = df.text.apply(preprocess)

#label
neg = df.loc[df['label'] == 'negative'].text_clean.tolist()
neu = df.loc[df['label'] == 'neutral'].text_clean.tolist()
pos = df.loc[df['label'] == 'positive'].text_clean.tolist()
neg_label = df.loc[df['label'] == 'negative'].label.tolist()
neu_label = df.loc[df['label'] == 'neutral'].label.tolist()
pos_label = df.loc[df['label'] == 'positive'].label.tolist()
total_data = pos + neu + neg
labels = pos_label + neu_label + neg_label

#Feature Extraction
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(total_data)
X = tokenizer.texts_to_sequences(total_data)
vocab_size = len(tokenizer.word_index)
maxlen = max(len(x) for x in X)
X = pad_sequences(X)
Y = pd.get_dummies(labels)
Y = Y.values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

sentiment = ['negative', 'neutral', 'positive']

#Load Model
import pickle
with open("models/mlp/feature.p", "rb") as file:
  count_vect = pickle.load(file)
with open("models/mlp/model.p", "rb") as file:
  model_mlp = pickle.load(file)

async def get_sentiment(input, type):
    if type == 'mlp':
        original_text = input
        text = count_vect.transform([preprocess(original_text)])
        result = model_mlp.predict(text)[0]
        return result
    else:
        try:
            if type == 'lstm':
                model = load_model('models/lstm/model_lstm.h5')
            else:
                model = load_model('models/rnn/model_rnn.h5')
                              
            input_text = input
            text = [preprocess(input_text)]
            predicted = tokenizer.texts_to_sequences(text)
            guess = pad_sequences(predicted, maxlen=X.shape[1])
            prediction = model.predict(guess)
            polarity = np.argmax(prediction[0])
            # sentiment = ['negative', 'neutral', 'positive']
            return sentiment[polarity]
        except Exception as e:
            print(e)


# async def get_sentiment_file(file, model):
#     if(model =="lstm"):
#         model = load_model('models/lstm/model_lstm.h5')
#     elif(model == "rnn"):
#         model = load_model('models/rnn/model_rnn.h5')
#     first_column = file.iloc[:, 0]
#     file = first_column.astype("string").apply(preprocess)
#     print("======== finish preprocess =========")

#     file = file.to_frame()
#     if(isinstance(file, pd.DataFrame)):
#         file.rename(columns={ file.columns[0]: "Tweet" }, inplace = True)
#         file["Sentiment"] = None
#         file['Tweet'] = file['Tweet'].astype('string')
#         file['Sentiment'] = file['Sentiment'].astype('string')

#         for i in range(len(file)):
#             text = file['Tweet'][i]
#             text = [text]
#             predicted = tokenizer.texts_to_sequences(text)
#             guess = pad_sequences(predicted, maxlen=X.shape[1])
#             prediction = model.predict(guess)
#             polarity = np.argmax(prediction[0])
#             file["Sentiment"][i] =  sentiment[polarity]

#         print("======== FINISH TEST =========")
#         return file
#     else:
#         print("======== FAILED TEST =========")
#         return "File is Unreadable"



async def get_sentiment_file(input, type):
    if type == 'mlp':
        original_text = input.loc[0, 'text']
        text = count_vect.transform([preprocess(original_text)])
        result = model_mlp.predict(text)[0]
        return original_text, result
    else:
        try:
            if type == 'rnn':
                model = load_model('models/rnn/model_rnn.h5')
            else:
                model = load_model('models/lstm/model_lstm.h5')

            input_text = input.loc[0, 'text']
            text = [preprocess(input_text)]
            predicted = tokenizer.texts_to_sequences(text)
            guess = pad_sequences(predicted, maxlen=X.shape[1])
            prediction = model.predict(guess)
            polarity = np.argmax(prediction[0])
            # sentiment = ['negative', 'neutral', 'positive']

            return input_text, sentiment[polarity]
        except Exception as e:
            print(e)