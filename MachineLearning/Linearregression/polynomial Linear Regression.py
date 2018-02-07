# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:08:58 2017

Linear regression for polynomials


@author: Michal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.close('all')

#%% 
#useful functions

def computePrediction(X, theta):
    """
    compute y for a given X and theta: h=X*theta
    """
    return np.dot(X, theta)



def computeCost(X, y, theta):
    """
    compute 'cost' of fitting: J=((h-y)^2) / (2*m)
    """
    return np.sum(((computePrediction(X,theta) - y) ** 2) / (2*m))

def computeCost(X, y, theta):
    """
    compute 'cost' of fitting with regularization:
    J=((h-y)^2 + lambda) / (2*m) 
    """
    return np.sum(((computePrediction(X,theta) - y) ** 2) / (2*m))


def gradientDescent(X, y, theta, alpha, maxIterNum, treshold):
    """
    this function looks for best theta parameters, using gradient descent
    """
    
    J_overTime = []
    J_new = computeCost(X, y, theta)
    J_old = J_new * 1.5
    numIter = 0 
    
    while ((J_old - J_new) / J_old > treshold) and (numIter < maxIterNum):
#        print('hello')
        J_old = J_new.copy()                
        theta_old = theta.copy()
        for par in range(n):
           theta[par] = theta_old[par] - (alpha/m) * np.dot(np.transpose(
                    computePrediction(X,theta_old)-y), X[:,par])
            
        J_new = computeCost(X, y, theta)            
        J_overTime.append(J_new)
        
        numIter = numIter + 1
     
    return theta, J_overTime, numIter


#def featureScaling(X):
    
def featureScaling(X,mean=None, std=None):
    """
    scale inputs to zero-mean and 1 std
    """
    
    if mean is None:
        mean = X.mean(axis=0).reshape(1, n-1)
    if std is None:
        std = X.std(axis=0).reshape(1, n-1)
    
    return (X-mean)/std, mean, std









#%%
#generating data

m = 500 # number of samples to generate

#equation: y = a0 + a1x1 + a2x1x2 + a3(x3)^3

theta_target = [18, 5, 1, 8] #contains polynomial coefficients
x0 = np.ones(m)
x1 = np.linspace(-5,5,m)
x2 = np.linspace(0,10,m)
x3 = np.linspace(4,-1,m)
noise = np.random.randn(m)*50

X_gen = np.stack((x0, x1, x1*x2, x3**3), axis=1)
data = np.stack((x1, x2, x3), axis=1)

#y = np.dot(X_gen,theta_target) + noise   #target
y = (computePrediction(X_gen, theta_target)+noise).reshape(m,1)

plt.figure('orgData')
plt.scatter(range(m),y,label='target')
plt.title('input data')
plt.xlabel('sample')
plt.ylabel('target')
plt.legend()


#%% preprocessing data

degree = 4 #maximum degree of fitting polynomial
powers = []
X = []
#creating all possible combinations of x 
# x1*x2*x3, x1^2*x2*x3, x1*x2^2*x3, x1*x2*x3^2, ...  

for i in range(degree+1):
    for j in range(degree+1):
        for k in range(degree+1):
            temp = data[:,0]**i * data[:,1]**j * data[:,2]**k
#            X = np.hstack((X,temp))
            X.append(temp)
            powers.append((i,j,k))
            
X = np.transpose(X)
            
#%%            

#X, X_test, y, y_test = train_test_split(X,y,train_size=1)
(m,n) = X.shape

#X, mean, std = featureScaling(X[:,1:])
#X = np.insert(X, 0, np.ones(m), axis=1)

#X_test, _,_ = featureScaling(X_test[:,1:], mean, std)
#X_test = np.insert(X_test, 0, np.ones(len(y_test)), axis=1)

#%% fitting data and estimating best theta values

#initialize fitting parameters
theta = np.zeros((n,1)) # theta represents fitting coefficients
alpha = 0.01
maxIterNum = 5000
treshold = 1e-11

(theta, J_overTime, numIter) = gradientDescent(X, y, theta, alpha, maxIterNum,
                                               treshold)

print('Best theta values are: ', theta)


# Cost function over time visualization
plt.figure('plt_J')
plt.plot(list(range(numIter)),J_overTime)
plt.title('cost function over time')
plt.xlabel('# of iterations')
plt.ylabel('value of cost function')


plt.figure('results')
y_predicted = computePrediction(X,theta)
plt.scatter(range(m),y_predicted)



#%%comparison to sklearn 

from sklearn.linear_model import LinearRegression

LR = LinearRegression(fit_intercept=True,normalize=True)
LR.fit(X,y)
coef = LR.coef_
coef_intercept = LR.intercept_
coef[0,0]=coef_intercept
thetaSKL = np.transpose(coef)

print('sklearn coefficients are: ',thetaSKL)

y_SKL = LR.predict(X) - coef_intercept#
plt.figure('sklearn')
plt.scatter(range(m),y_SKL)
#y_model = computePrediction(X_test, theta)

#Mean Square Error
#MSE_SKL = np.sum((y_test-y_SKL)**2) / len(y_test)
#MSE_model = np.sum((y_test-y_model)**2) / len(y_test)

#print('MSE for sklearn prediction is {0:.4f}, when for my model MSE is {1:.4f}'.
#      format(MSE_SKL, MSE_model))


