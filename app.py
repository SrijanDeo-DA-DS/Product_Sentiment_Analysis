from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import pandas as pd
import sklearn
import inflect
import re
import string
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import spacy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

app = Flask(__name__)
text_preprocess = pickle.load(open('text_preprocess.pkl','rb'))
cv = pickle.load(open('text_representation.pkl', 'rb'))
model = pickle.load(open('rf_model.pkl', 'rb'))


## Route for a home page

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        product_review = request.form['product_review']
        
        #product_review = ''.join(product_review)

        product_review = text_preprocess.transform(product_review)
        
        a=[]
        for i in product_review:
            a.append(''+i.text+'')

        count_vectorizer = cv.transform(a)

        prediction = model.predict(count_vectorizer)

        output = prediction[0]

        return render_template('home.html', results=output)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=0000)