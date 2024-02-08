import pandas as pd
import numpy as np
    
df = pd.read_csv(r'C:/Users/srija/GitHub Repositories/Multi_class-Sentiment_analysis/Processed_data.csv')

## Count total number of words in Corpus

total = 0
for i in df['Summary']:
    length = len(i)
    total = total + length
print("We have {} number of words in our corpus".format(total))


## Count total number of unique words in corpus (Vocabulary)

unique_words = []

for i in df['Summary']:
    if i not in unique_words:
        unique_words.append(i)
print("We have {} number of unique words.".format(len(unique_words)))

## Train test split

X = df.drop(['Sentiment'],axis=1)
y = df['Sentiment']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40,stratify=y)

## Output classes
list(le.classes_)


## BAG OF WORDS

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

bow = cv.fit_transform(X_train['Summary'])

print(cv.vocabulary_)

## Applying Bag Of Words

X_train_bow = cv.fit_transform(X_train['Summary']).toarray()
X_test_bow = cv.transform(X_test['Summary']).toarray()

import pickle

pickle.dump(cv,open('text_representation.pkl','wb'))


## Using RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rf_bow = RandomForestClassifier()

rf_bow.fit(X_train_bow,y_train)

y_pred = rf_bow.predict(X_test_bow)

## Check baseline accuracy and classification report
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))


## Using SMOTE to balance the dataset

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_oversample, y_oversample = oversample.fit_resample(X_train_bow,y_train)   

## RANDOM FOREST over oversampled data

rf_bow_smote = RandomForestClassifier(n_jobs=-1)
rf_bow_smote.fit(X_oversample,y_oversample)
y_pred = rf_bow_smote.predict(X_test_bow)

print(classification_report(y_test,y_pred))

## HYPERPARAMTER TUNING

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 250, num = 20)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5,10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2,4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)

rf_randomcv_smote=RandomizedSearchCV(estimator=rf_bow_smote,param_distributions=param_grid,n_iter=2,cv=2,verbose=2,
                               random_state=100,n_jobs=-1)
### fit the randomized model
rf_randomcv_smote.fit(X_oversample,y_oversample)

y_pred = rf_randomcv_smote.predict(X_test_bow)

print(classification_report(y_test,y_pred))
print(rf_randomcv_smote.best_params_)

pickle.dump(rf_randomcv_smote,open('rf_model.pkl','wb'))

print(rf_bow_smote.predict(cv.transform(['I have seen better item than this'])))