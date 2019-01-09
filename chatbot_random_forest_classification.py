#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 01:14:57 2019

@author: subham
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset1 = pd.read_csv('training_queries.csv')
dataset2 = pd.read_csv('training_queries_labels.csv')
#dataset3 = pd.read_csv('testing_queries.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
clean_query = []
for i in range(0, 1808):
    query = re.sub('[^a-zA-Z]', ' ', dataset1['query'][i])
    query = query.lower()
    query = query.split()
    ps = PorterStemmer()
    #query = [ps.stem(word) for word in query if not word in set(stopwords.words('english'))]
    for index,words in enumerate(query):
        if words in stopwords.words('english'):
            query.remove(words)
        else:
            stemed_word = ps.stem(words)
            query[index] = stemed_word
    query = ' '.join(query)
    clean_query.append(query)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(clean_query).toarray()
y = dataset2.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 500/1808)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy')
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
