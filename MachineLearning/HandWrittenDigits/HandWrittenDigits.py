# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:24:26 2017

@author: Michal

sklearn tutorial
hand written digits classification with LogisticRegression and KNeighborsClassifier
"""
from sklearn import datasets, neighbors, linear_model, model_selection
import pandas as pd

digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y)


#KNeighborsClassifier
knn = neighbors.KNeighborsClassifier()
knn_score = knn.fit(X_train, y_train).score(X_test, y_test)


#LogisticRegression
regr = linear_model.LogisticRegression()
regr_score = regr.fit(X_train, y_train).score(X_test, y_test)

print('Score of knn is {} when score of regr is {}'.format(knn_score,regr_score))
