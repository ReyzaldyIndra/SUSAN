import numpy as np
import pandas as pd
import spacy
import pickle
import re
import string
from flask import Flask, abort, jsonify, request
from spacy.lang.id.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from spacy.lang.id.stop_words import STOP_WORDS
from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))
nlp = spacy.blank('id')
app = Flask(__name__)

import re
def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z-0-9 ]+', '', text)
    text = _removeNonAscii(text)
#     print(text)
    text = text.strip()
#     text = text.split()
#     print(text)
    text = ''.join([nlp(word)[0].lemma_ for word in text])
#     print(text)
    return text

@app.route('/',methods=['POST','GET'])
@app.route('/test',methods=['POST','GET'])
def prediksi():
    sentences =request.args.get('sentences')
    df = pd.read_excel('chat bpjs v1.xlsx',sheet_name='Read_Conversation')
    df = df.drop(columns=['Conversation','Kata','Pertanyaan.1','NER.1','Jawaban.1','Entitas.1','Entitas_Kata (NER)'])
    # df = df.drop_duplicates(subset='Pertanyaan', keep="first").reset_index(drop=True)
    # df.loc[209,'Klasifikasi']='PEMESANAN'
    # df=df.dropna()
    df1 = df[['Pertanyaan','INTENT']]
    df2 = df[['Jawaban','INTENT']]
    df2.columns=['Pertanyaan','INTENT']
    df = pd.concat([df1,df2]).drop_duplicates(keep='first').reset_index(drop=True)
    inputs = df.Pertanyaan.values.tolist()
    labels = df.INTENT.values.tolist()
    inputs = [clean_text(text) for text in inputs]

    df_vect = TfidfVectorizer(max_features=5000)
    df_vect.fit(inputs)

    model = pickle.load(open("bpjs_svm.sav", "rb"))

    intent = np.array(['CLOSINGS', 'GREETINGS', 'OTHERS', 'PROFIL', 'RECORD',
        'TRANSACTION'])
    # sentences = "dimana provinsi saya terdaftar?"
    sentences = [clean_text(sentences)]
    sentences = df_vect.transform(sentences)
    hasil = model.predict(sentences)
    print(intent[hasil[0]])
    return intent[hasil[0]]
if __name__ == '__main__':
	print("Loading PyTorch model and Flask starting server ...")
	print("Please wait until server has fully started")
	app.run()