# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:58:58 2017

Implemenation of linear regression


@author: Michal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%% defining functions

def scale(x,mean=None, std=None):
    """
    scale inputs to zero-mean and 1 std \n
    """
    
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    
    return x.subtract(mean,axis=1).divide(std,axis=1), mean, std
    

def regression(x, theta):
    """
    returns product of X and theta - y of regression
    """
    return np.dot(x,theta)
    
    
def costFunction(x,y,theta):
    """
    returns Mean Square Error of regression \n
    """
    
    return np.sum((regression(x,theta) - y)**2) / (2*m)

    
def gradientDescent(alpha, theta, x, y):
    """
    calculate new theta values
    """
    
    theta_old = theta
    for par in range(n):
        theta[par] = theta_old[par] - (alpha/m) * np.dot(np.transpose(
                regression(x,theta)-y), x[:,par].reshape(m,1))
    
    return theta
        
def R2(y_predicted, y_test):
    """
    returns a sum of square errors
    """
    return np.sum((y_test - y_predicted)**2) / (2*y_test.size)


#%% loading and cleaning data
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

#decrease amount of data
#X = X.loc[:,['AGE']]
#y = y[:]

#N/A detecting and removing
if pd.isnull(X).any(axis=1).any(axis=0):
    X.dropna(axis=1, inplace=True)
    y = y[X.index]
if pd.isnull(y).any():
    y.dropna(inplace=True)
    X = X[y.index]


X_train, X_test, y_train, y_test = train_test_split(X, y)

#usefull variables
X_train_vis = X_train.values.copy()
X_test_vis = X_test.values.copy()
(m , n) = X_train.shape

#scalling features
(X_train, mean, std) = scale(X_train)
X_train.insert(loc=0,column='xs for intercept parameter', value=np.ones(m))
n = n+1

# go to numpy - easier to operate
X_train = X_train.values.reshape((m,n))
y_train = y_train.values.reshape((m,1))





#%%    determinig right theta
#initializing starting values
theta = np.ones((n,1))
iterationCounter = 0
mse = costFunction(X_train,y_train,theta) #Mean Square Error as a product of CostFunction
mseOverIteration = [mse] # for presentation purposes
alpha=0.01
mse_new = mse*0.9

#looping until converage
while (abs(mse-mse_new)/mse > 0.0001) and (iterationCounter <500) :
    mse = mse_new
    theta = gradientDescent(alpha, theta, X_train, y_train)
    mse_new = costFunction(X_train, y_train,theta)
    iterationCounter = iterationCounter + 1
    mseOverIteration.append(mse_new)

#estimation and prediction
y_estimated = np.dot(X_train,theta) # for visualization purposes

#data never seen before
#scalling and additing 1s column
X_test,_,_ = scale(X_test,mean,std)
X_test.insert(loc=0,column='xs for intercept parameter', value=np.ones(y_test.size))
#to numpy
X_test = X_test.values.reshape((y_test.size,n))
y_test = y_test.values.reshape((y_test.size,1))

y_predicted = np.dot(X_test, theta)



#%%comparison to sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_sk = lr.predict(X_test)
y_sk_estimated = lr.predict(X_train)

R2_mine = R2(y_predicted,y_test)
R2_sk = R2(y_sk,y_test)

print('The mean square error for my model is {0:.2f}, when sklearn result is {1:.2f}.'\
      'The difference is around {2:.2f}%'.format(R2_mine,R2_sk,(R2_mine-R2_sk)/R2_sk*100))
print('Theta values of my model are {}, when sklearn calculated {}'.format(
        theta,lr.coef_))

#%%    visualization
plt.close('all')

#mse over time
plt.figure(0)
plt.plot(mseOverIteration)
plt.xlabel('# of iteration')
plt.ylabel('value of cost function')

if n == 2: #visualization only for univariate linear regression
    #data point and linear regression
    plt.figure(1)
    
    #train and test data
    plt.scatter(X_train_vis,y_train, label='train data', color = 'g')
    plt.scatter(X_test_vis,y_test, label='test data', color = 'b')
    
    #regression line
    plt.plot(X_train_vis,y_estimated, label='regression line', color='k')
    
    #predicted points
    plt.scatter(X_test_vis,y_predicted, label='predicted values', color='k')
    
    #sklearn results
    plt.scatter(X_test_vis, y_sk, label='sklearn points', color='r')
    plt.plot(X_train_vis, y_sk_estimated, label='sklearn regression line', color='r')
    
    plt.xlabel(X.columns[0])
    plt.ylabel('target')
    plt.legend()

X.add()