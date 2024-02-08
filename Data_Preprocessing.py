import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import string


## Importing only 20k rows since less RAM
df = pd.read_csv('Dataset-SA.csv',nrows=20000)

print(df['Sentiment'].value_counts())

## This is an imbalanced dataset
df = df[['Summary','Sentiment']]

df.dropna(axis=0,inplace=True)


## We see there are integers in our data. We will convert them to string, i.e. 2 --> two

import inflect

def convert_numbers_to_words(input_string):
    words = input_string.split()
    p = inflect.engine()

    for i in range(len(words)):
        if words[i].isdigit():
            words[i] = p.number_to_words(words[i])

    return ' '.join(words)

df['Summary'] = df['Summary'].astype(str)
df['Summary'] = df['Summary'].apply(convert_numbers_to_words)

## New column to count the length of reviews
df['Summary_len'] = df['Summary'].apply(lambda i:len(i.split(" ")))

## Review length vs Sentiment
sns.histplot(data=df,x='Summary_len',bins=20)

df.groupby('Sentiment')['Summary_len'].median().plot(kind='bar')
plt.ylabel('Length of review')

## TEXT CLEANING

# 1. Converting all to lowercase

#df['Summary'] = df['Summary'].apply(lambda i:i.lower())

# 2. Remove HTML tags if any

def html_tags_remove(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'',text)

df['Summary'] = df['Summary'].apply(html_tags_remove)

# 3. Remove URLs if any

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)

df['Summary'] = df['Summary'].apply(remove_url)

# 4. Remove Punctuations

exclude = string.punctuation

def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))

df['Summary'] = df['Summary'].apply(remove_punc)

# 5. Spelling Check
from textblob import TextBlob
def correct_text(text):
    txtblb = TextBlob(text)
    return txtblb.correct().string
df['Summary'] = df['Summary'].apply(correct_text)

# 6. Remove Stopwords
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def remove_stopwords(text):
    
    new_lst = []
    
    for i in text.split():
        if i not in stopwords.words('english'):
            new_lst.append(i)
            
    x = new_lst[:]
    new_lst.clear()
    return " ".join(x)

df['Summary'] = df['Summary'].apply(remove_stopwords)

# 7. Stemming
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem_words(text):
    return " ".join([ps.stem(i) for i in text.split()])

df['Summary'] = df['Summary'].apply(stem_words)

# 8. Using SPACY to tokenize
import spacy
nlp = spacy.load('en_core_web_sm')

def tokenize(text):
    doc = nlp(text)
    
    lst = []
    for i in doc:
        lst.append(i)
    
    return lst

df['Summary'] = df['Summary'].apply(tokenize)

df.to_csv('Processed_data.csv',index=False)

## Create Pipelines for Test data

from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.preprocessing import FunctionTransformer

trf1 = FunctionTransformer(convert_numbers_to_words)
trf2 = FunctionTransformer(html_tags_remove)
trf3 = FunctionTransformer(remove_url)
trf4 = FunctionTransformer(remove_punc)
trf5 = FunctionTransformer(remove_stopwords)
trf6 = FunctionTransformer(stem_words)
trf7 = FunctionTransformer(tokenize)

sk_pipe = Pipeline([("trans1", trf1),("trans2", trf2),("trans3", trf3),("trans4", trf4),
                   ("trans5", trf5),("trans6", trf6),("trans7", trf7)])


sk_pipe.transform('I have seen better item than this')

## Pickle the current Pipeline

import pickle

pickle.dump(sk_pipe,open('text_preprocess.pkl','wb'))