# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:24:26 2017

@author: Michal

sklearn tutorial
hand written digits classification with LogisticRegression and KNeighborsClassifier
"""
from sklearn import datasets, neighbors, linear_model, model_selection, svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

digits = datasets.load_digits()
X = pd.DataFrame(digits.data)
y = pd.Series(digits.target)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y)


#testing crossvalidatin score as a function of the C parameter for SVC
C_range = np.logspace(-10,0,10)
svc = svm.SVC(kernel='linear')
val_score = list()
svc_score = list()

for C in C_range:
    svc.C = C
    val_score.append(model_selection.cross_val_score(svc,X_train, y_train).mean())
    svc_score.append(svc.fit(X_train, y_train).score(X_test, y_test))

plt.plot(C_range,val_score)
plt.semilogx()
plt.title('dependency of crossvalidation score as a function of the C parameter')
plt.xlabel('C value')
plt.ylabel('crossvalidation score')

#KNeighborsClassifier    
knc = neighbors.KNeighborsClassifier()
knc_score = knc.fit(X_train, y_train).score(X_test, y_test)

#LogisticRegression
regr = linear_model.LogisticRegression()
regr_score = regr.fit(X_train, y_train).score(X_test, y_test)

print('Score of knn is {} when score of regr is {} and score of svc is {}'.format(
        knc_score,regr_score, np.max(svc_score)))
