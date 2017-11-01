# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:17:35 2017

@author: Michal

Comparison of different classifiers
Idea is to create a model wihch will decide whether particular muschroom is edible,
based on physical properties
data available at #https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cvs

#loading data
X = pd.read_csv('agaricus-lepiota.data', header=0, na_values='?')

#removing N/A
if pd.isnull(X).any(axis=1).any(axis=0):
    X.dropna(axis=0, inplace=True)

#separating targets from data
y = X.loc[:,'edible'].copy()
X.drop('edible', axis=1, inplace=True)

#encoding textual features into numerical ones
y = y.map({'e':1, 'p':0})
X = pd.get_dummies(X)

#splitting into train, test and validation sets
X_train_validation, X_test, y_train_validation, y_test = \
                                    train_test_split(X, y, test_size=0.15)
                                                                          

"""
model selection
based on sk-learn documentation, Linear Support Vector Classification should do
well here. However, for learning purposes, I will check and compare few more
"""
#comparison = pd.DataFrame(columns = ['model','CrossValidationScore'])
#comparison = {} #model and Cross Valdiation Score
models = pd.Series()
scores = pd.Series(dtype=np.float)

#LinearSVC
from sklearn.svm import LinearSVC
models = models.append(pd.Series('LinearSVC()'), ignore_index=True)
    
#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
models = models.append(pd.Series('KNeighborsClassifier()'), ignore_index=True)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
models = models.append(pd.Series('DecisionTreeClassifier()'), ignore_index=True)


#crosvalidation
for model_name in models:
    model = eval(model_name)
    scores = scores.append(pd.Series(cvs(model ,X_train_validation, 
                        y_train_validation, cv=3).mean()*100), ignore_index=True)


#final test
print('Maksimum learning score is {0}% for {1}'.format(scores.max(),
      models[scores.idxmax()]))
model = eval(models[scores.idxmax()])
model.fit(X_test, y_test)
final_score = model.score(X_test, y_test)*100

print('when final score is {}%'.format(final_score))


# to do
# wizualizacja plus sieci neuronowe

















    