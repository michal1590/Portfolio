# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 01:39:18 2017

@author: Michal

Diabetes
Looking for best alpha parameter for Lasso estimator
"""

from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import numpy as np

diabetes = datasets.load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

lasso = Lasso()

gscv = GridSearchCV(lasso,{'alpha':np.logspace(-4,-0.5,30)})
gscv.fit(X_train, y_train)
gscv_score = gscv.cv_results_['mean_test_score']
best_alpha = gscv.best_params_['alpha']
print('best alpha value is {}'.format(best_alpha))

lasso.alpha = best_alpha
lasso.fit(X_train, y_train)
lasso_score = lasso.score(X_test, y_test)
print('best lasso score is {}'.format(lasso_score))
