# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:43:54 2017

my implementation of Linear Regression with one variable


@author: Michal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% some useful functions

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


def gradientDescent(X, y, theta, alpha, maxIterNum, treshold):
    """
    this function looks for best theta parameters, using gradient descent
    """
    
    J_overTime = []
    J_new = computeCost(X, y, theta)
    J_old = J_new * 1.5
    numIter = 0 
    
    while ((J_old - J_new) / J_old > treshold) and (numIter < maxIterNum):
        
        J_old = J_new.copy()                
        theta_old = theta.copy()
        for par in range(n):
           theta[par] = theta_old[par] - (alpha/m) * np.dot(np.transpose(
                    computePrediction(X,theta_old)-y), X[:,par])
            
        J_new = computeCost(X, y, theta)            
        J_overTime.append(J_new)
        
        numIter = numIter + 1
     
    return theta, J_overTime, numIter




#%% loading and preprocessing data

plt.close('all')

data = pd.read_csv(r'E:\Nauka\Machine Learning by Stanford University\week2\exercise\ex1\ex1data1.txt',
                   header=None, names=['Population','Profit'])


#data.dropna(axis='index',inplace=True)
#data.plot.scatter('Population','Profit')
#plt.title('raw data')
#data = data.iloc[:10,:]

(m,n) = data.shape
n = n-1 #because target column was included

#going to numpy for calculation part
X = data.Population.values.reshape((m,n))
y = data.Profit.values.reshape((m,1))

X = np.insert(X, 0, np.ones(m), axis=1) #interception column


X, X_test, y, y_test = train_test_split(X,y)
(m,n) = X.shape

"""
#some test points

theta = np.zeros((n,1))
J = computeCost(X, y, theta)
print('cost for theta = zeros',J)

theta[0] = -1
theta[1] = 2
J = computeCost(X, y, theta)
print('cost for theta =[-1,2]',J)
"""


#%% fitting data and estimating best theta values

#initialize fitting parameters
theta = np.zeros((n,1))
alpha = 0.01
maxIterNum = 5000
treshold = 1e-11

(theta, J_overTime, numIter) = gradientDescent(X, y, theta, alpha, maxIterNum,
                                               treshold)

print('Best theta values are: ', theta)


# Cost function over time visualization
plt_J = plt.figure()
plt.plot(list(range(numIter)),J_overTime)
plt.title('cost function over time')
plt.xlabel('# of iterations')
plt.ylabel('value of cost function')




#%%comparison to sklearn 

from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X,y)
coef = LR.coef_
coef_intercept = LR.intercept_
thetaSKL = np.array([coef_intercept,coef[:,1:]])

print('sklearn coefficients are: ', coef_intercept, ' and ', coef[:,1:])

y_SKL = LR.predict(X_test)
y_model = computePrediction(X_test, theta)

#Mean Square Error
MSE_SKL = np.sum((y_test-y_SKL)**2) / len(y_test)
MSE_model = np.sum((y_test-y_model)**2) / len(y_test)

print('MSE for sklearn prediction is {0:.4f}, when for my model MSE is {1:.4f}'.
      format(MSE_SKL, MSE_model))



#%% Visualization

dataAndLines = plt.figure()
feature = 1 # feature to scatter - col0 is an intercept column

plt.title('train and test data')
plt.ylabel('target')
plt.xlabel(data.columns[feature-1])


#train and test data
plt.scatter(X[:,feature],y,label='Train data', c='g', marker='o')

plt.scatter(X_test[:,feature], computePrediction(X_test,thetaSKL), 
            label='skelarn prediction', c='r', marker = '*', linewidths=5)
plt.scatter(X_test[:,feature], computePrediction(X_test,theta), 
            label='model prediction', c='b', marker = 'x')

#regression lines
linesX = np.stack((X.min(axis=0),X.max(axis=0)),0) 
plt.plot(linesX[:,feature], computePrediction(linesX,thetaSKL),label='skelarn regresion line',
         c='r', lw=5)

plt.plot(linesX[:,feature], computePrediction(linesX,theta),label='model regresion line',
         c='b')
plt.legend()


# more J visualization

theta0 = np.linspace(-10,10,100)
theta1 = np.linspace(-10,10,100)
Jxx, Jyy = np.meshgrid(theta0, theta1)
Z = np.ones(Jxx.shape)
thetaTemp = np.zeros(theta.shape)


for i in range(len(theta0)):
    for j in range(len(theta1)):
        thetaTemp[0] = theta0[i]
        thetaTemp[1] = theta0[j]
        Z[i,j] = computeCost(X, y, thetaTemp)




Z = np.transpose(Z)

J_cont = plt.figure()
plt.contour(Jxx, Jyy, Z, np.logspace(-2,3,num=15))
plt.scatter(theta[0],theta[1],marker='x', c='r', linewidths=50)
plt.scatter(thetaSKL[0],thetaSKL[1],marker='*', c='k', linewidths=1)
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Cost as a function of theta')
        


















