# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:56:56 2019

@author: jeffr
"""

#Artifical Neural Network

#Importing The Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing our datasets
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values
#.astype(np.int64)
y = dataset.iloc[:, 13].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])

labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x.astype('float64')

x = x[:,1:]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x =  StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Part 2 - Now let's make the ANN!

#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Initializing the ANN
classifier = Sequential()

#Adding the Input Layer and the first hidden layer with dropout
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))
#Adding the second Hidden Layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))
#Adding the output Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])

#Fitting the ANN to the Training Set
classifier.fit(x= x_train,y = y_train, batch_size = 10, nb_epoch = 100)


#Part 3 - Making the predictions and evaluating the roles


#Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > .5)

#Predicting a single new observation
"""Predict if new customer will leave the bank.
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40 years old
    Tenure: 3 years
    Balance: $60000
    Number of Products: 2
    Does this customer have a credit card ? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $50000"""

new_prediction = classifier.predict(sc_x.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > .5)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Part 4 - Evaluating, Improving, and Finetuning the ANN

#Evaluate the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
	classifier = Sequential()
	classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
	classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
	classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])
	return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = 1) 

mean = accuracies.mean()
variance = accuracies.std()

#Improve the ANN

#Dropout Regularization to reduce overfitting if needed

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
	classifier = Sequential()
	classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
	classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
	classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics= ['accuracy'])
	return classifier
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25,32], 
			  'epochs': [100,500],
			  'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
						   param_grid = parameters,
						   scoring = 'accuracy',
						   cv = 10)
grid_search = grid_search.fit(x_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_







