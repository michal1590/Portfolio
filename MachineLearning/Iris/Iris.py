# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:45:23 2017

@author: Michal

sklearn tutorial
iris clasification with SVM
"""

from sklearn import datasets, svm, model_selection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#loading data
plt.close('all')
data = datasets.load_iris()
iris = pd.DataFrame(data.data,columns=data.feature_names)
target = pd.Series(data.target)


#preparing data for first experiment(only two first features) 

X = iris.iloc[:,:2]
y = target.copy()


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.1)


# training with different kernels
for figNum, kernel in enumerate (('linear', 'rbf', 'poly')):
    
    svc = svm.SVC(kernel=kernel)
    
    svc.fit(X_train, y_train)
    svc_score = svc.score(X_test, y_test)
    
    print('Score of SVC with {} kernel is {}%'.format(kernel, np.int(svc_score*100)))
    
    ##visualization
    plt.figure(figNum)
    
    #ploting decision surface
    u = np.linspace(min(X.iloc[:,0])*0.9,max(X.iloc[:,0])*1.1, num=10)
    v = np.linspace(min(X.iloc[:,1])*0.9,max(X.iloc[:,1])*1.1, num=10)
    XX, YY = np.meshgrid(u,v) #mesh to create a decision surface
    
    converted_mesh = np.c_[XX.ravel(), YY.ravel()] # mesh in different format
    Z = svc.predict(converted_mesh) # calculating estimator response for each point
    Z = Z.reshape(XX.shape) # back to the initial format
    plt.contourf(XX, YY, Z, 2)
    
    #plotting point of decision surface
    plt.scatter(X[y==0].iloc[:,0], X[y==0].iloc[:,1], label = data.target_names[0])
    plt.scatter(X[y==1].iloc[:,0], X[y==1].iloc[:,1], label = data.target_names[1])
    plt.scatter(X[y==2].iloc[:,0], X[y==2].iloc[:,1], label = data.target_names[2])
    
    plt.xlabel(iris.columns[0])
    plt.ylabel(iris.columns[1])
    plt.legend(loc=2)
    plt.title('decision surface and points in feature space for {} kernel'.format(kernel))
    




