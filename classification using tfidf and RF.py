# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:18:56 2021

@author: aai00920
"""
"""
Steps for classification: 
                        Step1: text preproceesing and EDA
                        Step2: convert text to number i.e. vectorizing the text (earlier tfidf, countvectorizer now elmo,                                       bert etc)
                        Step3 : model definition: earlier we used logistic regression, random forest, Naive Bayes etc now we use deep learning models especially sequential deep learning models
                        Step4: Fit the model with training data
                        Step5: Validation of the model
                        Step6: Fine tune the model
"""


import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

os.chdir(r'C:\Users\AAI00920\OneDrive - ARCADIS\Desktop\NLP\classification')
ehs_data = pd.read_csv('Sample Dataset.csv')

#there is only one nan i.e. one paragraph is not classified into a class so removing the nan
#ehs_data[ehs_data['Nature of Injury'].isnull()]
ehs_data_1 = ehs_data[ehs_data['Nature of Injury'].notna()]

#step1 : understanding the classes
ehs_classes = ehs_data_1['Nature of Injury'].unique()
ehs_classval = pd.DataFrame(ehs_data_1['Nature of Injury'].value_counts())

#step 2: convert text to numbers
sentence = ehs_data['Abstract Text'][0]
sentence = sentence.lower()
cleanr = re.compile('<.*?>')
cleantext = re.sub(cleanr, '', sentence) # applying the above regex function and cleaning the entire text
rem_num = re.sub('[0-9]+', '', cleantext) # remove digits
tokenizer = RegexpTokenizer(r'\w+') # RegexpTokenizer splits a string into substrings using a regular expression.
tokens = tokenizer.tokenize(rem_num) # tokenizing the text using above function 
filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')] # removing stopwords [exception_0]
stem_words=[stemmer.stem(w) for w in filtered_words] # retrieves the base/root word [exception_1]
lemma_words=[lemmatizer.lemmatize(w) for w in stem_words] # extracting the context of the word [exception_2]
" ".join(filtered_words)

#step 2: convert to function

def preprocess(sentence):
    sentence = sentence.lower()
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence) # applying the above regex function and cleaning the entire text
    rem_num = re.sub('[0-9]+', '', cleantext) # remove digits
    tokenizer = RegexpTokenizer(r'\w+') # RegexpTokenizer splits a string into substrings using a regular expression.
    tokens = tokenizer.tokenize(rem_num) # tokenizing the text using above function 
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')] # removing stopwords [exception_0]
    stem_words=[stemmer.stem(w) for w in filtered_words] # retrieves the base/root word [exception_1]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words] # extracting the context of the word [exception_2]
    return " ".join(filtered_words)

# Applying the function to the column data
ehs_data_1['CleanText'] = ehs_data_1['Nature of Injury'].map(lambda s:preprocess(s)) 

# class distribution EDA
ehs_data_1.groupby('Nature of Injury').CleanText.count().plot.bar(ylim=0)

# TF-IDF (Term Frequency-Inverse Document Frequency) 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 3, sublinear_tf = True, ngram_range = (1, 2))
final_features = vectorizer.fit_transform(ehs_data_1['Nature of Injury']).toarray()
final_features.shape

#%%
#-----training the model using Random Forest Classifier-------#

# splitting dataset into train and test

from sklearn.model_selection import train_test_split
X = ehs_data_1['Abstract Text']
Y = ehs_data_1['Nature of Injury']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.1)

# creating a pipeline to set functions
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=750)),
                     ('clf', RandomForestClassifier())])

# Fitting the model and saving it
import pickle

model = pipeline.fit(X, Y)
with open('Classification_ehs.pickle', 'wb') as f:
    pickle.dump(model, f)

import numpy as np    
ytest = np.array(y_test)
ytrain = np.array(y_train)

#--------Evaluating the model-------#
# confusion matrix and classification report(precision, recall, F1-score)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(ytrain, model.predict(X_train)))
print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))

test_sentence = 'At approximately 1:30 p.m. on February 22, 2017, an alleged report was received of an employee being struck by a forklift while backing up was received. '

model.predict([test_sentence])






