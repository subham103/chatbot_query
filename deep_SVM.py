#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 00:27:14 2019

@author: subham
"""



# Query Classifier

# Importing the libraries
import numpy as np
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

#Converting dataset1(train data) to a list of stemed words
for i in range(0, 1808):
    query = re.sub('[^a-zA-Z]', ' ', dataset1['query'][i])
    query = query.lower()
    query = query.split()
    ps = PorterStemmer()
    for index,words in enumerate(query):
        if words in stopwords.words('english'):
            query.remove(words)
        else:
            stemed_word = ps.stem(words)
            query[index] = stemed_word
    query = ' '.join(query)
    clean_query1.append(query)

#Converting dataset3(test data) to a list of stemed words
for j in range(0, 425):
    query2 = re.sub('[^a-zA-Z]', ' ', dataset3['query'][j])
    query2 = query2.lower()
    query2 = query2.split()
    ps2 = PorterStemmer()
    for index,words in enumerate(query2):
        if words in stopwords.words('english'):
            query2.remove(words)
        else:
            stemed_word = ps2.stem(words)
            query2[index] = stemed_word
    query2 = ' '.join(query2)
    clean_query2.append(query2)
#Appending stemed words in for test data to train data
for k in range(425):
    clean_query1.append(clean_query2[k])
    
#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer()

#Creating a matrix (to count the number of occurance of a word in a perticular query) from the stemed word list(clean_query1) 
x = cv1.fit_transform(clean_query1).toarray()

#extracting the ground truth of the train data and reshaping it
y = dataset2.iloc[:, 1].values
y = np.reshape(y, (1808,1))

#Encoding the ground truth to dummy variable format
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray()

#Separating training and testing rows from the encoded matrix
x_test = x[1808: , :]
x = x[:1808, :]

#Creating an ANN model 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU

classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 727, init = 'uniform', input_dim = 1445))
classifier.add(LeakyReLU(alpha=0.05))

#Adding the second hidden layer
classifier.add(Dense(output_dim = 727, init = 'uniform'))
classifier.add(LeakyReLU(alpha=0.05))

#Adding the third hidden layer
classifier.add(Dense(output_dim = 727, init = 'uniform'))
classifier.add(LeakyReLU(alpha=0.05))

#Adding the fouth hidden layer
classifier.add(Dense(output_dim = 727, init = 'uniform'))
classifier.add(LeakyReLU(alpha=0.05))

#Adding the output layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN model to the Training set
classifier.fit(x, y, batch_size = 30, nb_epoch = 40)

#Predicting the Test set results and train set results
y_pred = classifier.predict(x_test)
x_pred = classifier.predict(x)

#Decoding traning ground truth(y (dummy)) to its original format
def decode(data):
    return np.argmax(data)

decoded_y = []
for j in range(1808):
    decoded_y.append(decode(y[j]))

#Creating a linear SVM model    
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'linear')

#Fitting the linear SVM model to the output of ANN model for training data(x_pred) and Ground truth(decoded_y)
classifier2.fit(x_pred, decoded_y)

#Predicting the tags for Test data 
y_final = classifier2.predict(y_pred)

#Extracting the index of test data
y_sub = dataset3.iloc[:, 0].values

#Writing the results for test data to a CSV file
file = open("deep_learning_and_SVM.csv", "w")
file.write("index,tag \n") 
for k in range(425):
    file.write(str(y_sub[k]) + "," + str(y_final[k]) + "\n")
    
file.close()
