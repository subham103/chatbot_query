#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 00:27:14 2019

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
dataset3 = pd.read_csv('test.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
clean_query1 = []
clean_query2 = []
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
    clean_query1.append(query)

for j in range(0, 425):
    query2 = re.sub('[^a-zA-Z]', ' ', dataset3['query'][j])
    query2 = query2.lower()
    query2 = query2.split()
    ps2 = PorterStemmer()
    #query = [ps.stem(word) for word in query if not word in set(stopwords.words('english'))]
    for index,words in enumerate(query2):
        if words in stopwords.words('english'):
            query2.remove(words)
        else:
            stemed_word = ps2.stem(words)
            query2[index] = stemed_word
    query2 = ' '.join(query2)
    clean_query2.append(query2)

for k in range(425):
    clean_query1.append(clean_query2[k])
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer()
#cv2 = CountVectorizer()


x = cv1.fit_transform(clean_query1).toarray()
y = dataset2.iloc[:, 1].values
y = np.reshape(y, (1808,1))

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray()
#x_test = cv1.fit_transform(clean_query2).toarray()

x_test = x[1808: , :]
x = x[:1808, :]

from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import LeakyReLU
from keras.layers.advanced_activations import LeakyReLU, PReLU

classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 727, init = 'uniform', input_dim = 1445))
classifier.add(LeakyReLU(alpha=0.05))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 727, init = 'uniform'))
classifier.add(LeakyReLU(alpha=0.05))
# Adding the third hidden layer
classifier.add(Dense(output_dim = 727, init = 'uniform'))
classifier.add(LeakyReLU(alpha=0.05))
# Adding the fouth hidden layer
classifier.add(Dense(output_dim = 727, init = 'uniform'))
classifier.add(LeakyReLU(alpha=0.05))

# Adding the output layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x, y, batch_size = 30, nb_epoch = 40)

# Part 3 - Making the predictions and evaluating the model
def decode(data):
    return np.argmax(data)
# Predicting the Test set results
y_pred = classifier.predict(x_test)
#y_pred = np.reshape(y_pred, (425,1))
x_pred = classifier.predict(x)
decoded_y = []
for j in range(1808):
    decoded_y.append(decode(y[j]))

from sklearn.svm import SVC
classifier2 = SVC(kernel = 'linear')
classifier2.fit(x_pred, decoded_y)

y_final = classifier2.predict(y_pred)
'''
for p1 in range(425):
    max = 0
    a = [0] * 10
    for p2 in range(10):
        if y_pred[p1][p2]>max:
            max = y_pred[p1][p2]
            temp = p2
    a[temp] = 1
    y_pred[p1] = a
    
#y_sub = np.reshape(y_sub, (425,1))

decoded_y_pred = []
for j in range(425):
    decoded_y_pred.append(decode(y_pred[j]))
'''

#y_pred = np.reshape(y_pred, (425,1))

y_sub = dataset3.iloc[:, 0].values
#y_sub = np.reshape(y_sub, (425,1))

file = open("deep_learning_and_SVM.csv", "w")
file.write("index,tag \n") 
for k in range(425):
    file.write(str(y_sub[k]) + "," + str(y_final[k]) + "\n")
    
file.close()
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
