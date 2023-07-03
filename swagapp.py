#import module
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import re 
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pathlib import Path
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))

app = Flask(__name__)

#load Data
df= pd.read_csv('D:/Binar/Challenge/Binarchallenge2/tugas_dsc/data/databaru.txt', header=None)
df.columns =['text', 'label']

#Cleansing
factory = StemmerFactory()
stemmer = factory.create_stemmer()

alay_dict = pd.read_csv('data/new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})

id_stopword_dict = pd.read_csv('data/stopwordbahasa.csv', header=None)
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})

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

def remove_emoji(text):
    subs = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    text_emo = subs.sub(r'',text)
    return text_emo

alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

def remove_stopword(text):
    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def stemming(text):
    return stemmer.stem(text)

def preprocess(text):
    text = lowercase(text) 
    text = remove_nonaplhanumeric(text) 
    text = remove_unnecessary_char(text) 
    text = remove_emoji(text)
    text = normalize_alay(text) 
    text = stemming(text) 
    text = remove_stopword(text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)

    return text

# swagger app
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling'),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)

#feature extraction
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ',lower=True)
sentiment = ['negative', 'neutral', 'positive']

# tokenizer rnn
file_rnn = open('models/rnn/x_pad_sequences.pickle','rb')
feature_file_from_rnn = pickle.load(file_rnn)
file_rnn.close()

# tokenizer lstm
file_lstm = open('models/lstm/x_pad_sequences.pickle','rb')
feature_file_from_lstm = pickle.load(file_lstm)
file_lstm.close()

# load model 
model_rnn = load_model('models/rnn/model_rnn.h5')
model_lstm = load_model('models/lstm/model_lstm.h5')

# Endpoint RNN teks
@swag_from('docs/rnn_text.yml',methods=['POST']) 
@app.route('/rnn_text',methods=['POST'])
def rnn_text():

    upload_text = request.form.get('text')

    text = [preprocess(upload_text)]

    feature = tokenizer.texts_to_sequences(text)
    rnn_test = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
    
    prediction = model_rnn.predict(rnn_test)
    polarity = np.argmax(prediction[0])
    get_sentiment = sentiment[polarity]

    json_response = {
        'status_code': 200,
        'description': "Hasil sentimen analisis dengan model RNN",
        'data': {
            'text': text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint rnn file
@swag_from('docs/rnn_file.yml',methods=['POST'])
@app.route('/rnn_file',methods=['POST'])
def rnn_file():
    file = request.files["upload_file"]
    df = pd.read_csv(file, encoding='latin-1')
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row : preprocess(row['text']), axis = 1)
    
    result = []

    for index, row in df.iterrows():
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        rnn_test = pad_sequences(text, maxlen=feature_file_from_rnn.shape[1])
        prediction = model_rnn.predict(rnn_test)
        polarity = np.argmax(prediction[0])
        get_sentiment = sentiment[polarity]
        result.append(get_sentiment)

    original = df.text_clean.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Hasil sentimen analisis dengan model RNN",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint LSTM teks
@swag_from('docs/LSTM_text.yml',methods=['POST'])
@app.route('/LSTM_text',methods=['POST'])
def lstm_text():

    upload_text = request.form.get('text')

    text = [preprocess(upload_text)]

    feature = tokenizer.texts_to_sequences(text)
    lstm_test = pad_sequences(feature,maxlen=feature_file_from_lstm.shape[1])
    
    prediction = model_lstm.predict(lstm_test)
    polarity = np.argmax(prediction[0])
    get_sentiment = sentiment[polarity]

    json_response = {
        'status_code': 200,
        'description': "Hasil sentimen analisis dengan model RNN",
        'data': {
            'text': text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint LSTM file
@swag_from('docs/LSTM_file.yml',methods=['POST'])
@app.route('/LSTM_file',methods=['POST'])
def lstm_file():
    file = request.files["upload_file"]
    df = pd.read_csv(file, encoding='latin-1')
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row : preprocess(row['text']), axis = 1)
    
    result = []

    for index, row in df.iterrows():
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        lstm_test = pad_sequences(text, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_lstm.predict(lstm_test)
        polarity = np.argmax(prediction[0])
        get_sentiment = sentiment[polarity]
        result.append(get_sentiment)

    original = df.text_clean.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Hasil sentimen analisis dengan model RNN",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()


# --- end swagger ---