from flask import Flask, request, jsonify
import re
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from tensorflow.keras.initializers import Orthogonal

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

indo_slang_word = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
normalization_dict = dict(zip(indo_slang_word['slang'], indo_slang_word['formal']))

def normalize_text(text):
    words = word_tokenize(text)
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def clean_text(text):
    text = text.lower()
    text = normalize_text(text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Custom objects
custom_objects = {'Orthogonal': Orthogonal}

# Load the model with custom objects
model = tf.keras.models.load_model('model/analysis_sentiment_model.h5', custom_objects=custom_objects)
print("Model loaded successfully")
print("Model input shape:", model.input_shape)

tokenizer = Tokenizer()

@app.route('model/sentimentAnalysis', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    preprocessed_text = clean_text(text)
    
    tokenizer.fit_on_texts([preprocessed_text])
    sequences = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequences = pad_sequences(sequences, maxlen=16)  # Use the expected input length

    prediction = model.predict(padded_sequences)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
