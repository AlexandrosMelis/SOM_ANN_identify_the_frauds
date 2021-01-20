# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:59:42 2021
@author: Melis

Hybrid Deep Learning Model
"""

# Part 1: Identify the frauds with the Self-organizing map

"""### Importing the libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Importing the dataset"""

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

"""## Feature Scaling"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

"""##Training the SOM"""

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

"""##Visualizing the results"""

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

"""## Finding the frauds"""

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,7)], mappings[(7,3)], mappings[(1,1)],mappings[(3,6)]), axis = 0)
frauds = sc.inverse_transform(frauds)

"""##Printing the Fraunch Clients"""

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))


# Part 2: Predict the propability that each customer cheated with ANN

#creating the matrix of features
customers = dataset.iloc[:, 1:].values 

#creating the dependent variable
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

"""## Importing the keras libraries and packages"""
from keras.models import Sequential
from keras.layers import Dense

"""## Initializing the ANN"""
classifier = Sequential()

"""## Adding the input layer and the first hidden layer"""
classifier.add(Dense(units= 4, kernel_initializer = 'uniform',activation = 'relu', input_dim = 15))

"""## Adding the output layer"""
classifier.add(Dense(units= 1, kernel_initializer = 'uniform',activation = 'sigmoid'))

"""## Compiling the ANN"""
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""## Fitting the ANN to the training set"""
classifier.fit( customers, is_fraud, batch_size = 1, epochs = 2)


"""## Predicting the propabilities of frauds"""
y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

#Sort the array by one column i have chosen
y_pred = y_pred[y_pred[:,1].argsort()]
