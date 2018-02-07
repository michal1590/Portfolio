# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:43:54 2017

my implementation of Linear Regression


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


#def featureScaling(X):
    
def featureScaling(X,mean=None, std=None):
    """
    scale inputs to zero-mean and 1 std
    """
    
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    
    return (X-mean)/std, mean, std




#%% loading and preprocessing data

plt.close('all')


# estimation of company profits as a function of city size
#data = pd.read_csv(r'E:\Nauka\Machine Learning by Stanford University\week2\exercise\ex1\ex1data1.txt',
#                   header=None, names=['Population','Profit'])


#boston house-prices dataset - price estimation
from sklearn.datasets import load_boston
boston = load_boston()

data = pd.DataFrame(boston.data,columns=boston.feature_names)
data['target'] = boston.target

"""
#NA removing and basic data presentation
data.dropna(axis='index',inplace=True)
data.plot.scatter('Population','Profit')
plt.title('raw data')
data = data.iloc[:10,:]
"""
(m,n) = data.shape
n = n-1 #because target column was included

#going to numpy for calculation part
X = data.iloc[:,:-1].values.reshape((m,n))
y = data.iloc[:,-1].values.reshape((m,1))

#X = np.insert(X, 0, np.ones(m), axis=1) #interception column


X, X_test, y, y_test = train_test_split(X,y)
(m,n) = X.shape

X, mean, std = featureScaling(X)
X_test, _,_ = featureScaling(X_test, mean, std)


X = np.insert(X, 0, np.ones(m), axis=1) #interception column
n = n + 1
X_test = np.insert(X_test, 0, np.ones(len(y_test)), axis=1) #interception column

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
theta = np.zeros((n,1)) # theta represents fitting coefficients
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

LR = LinearRegression(fit_intercept=True)
LR.fit(X,y)
coef = LR.coef_
coef_intercept = LR.intercept_
coef[0,0]=coef_intercept
thetaSKL = np.transpose(coef)

print('sklearn coefficients are: ',thetaSKL)

y_SKL = LR.predict(X_test) - coef_intercept #
y_model = computePrediction(X_test, theta)

#Mean Square Error
MSE_SKL = np.sum((y_test-y_SKL)**2) / len(y_test)
MSE_model = np.sum((y_test-y_model)**2) / len(y_test)

print('MSE for sklearn prediction is {0:.4f}, when for my model MSE is {1:.4f}'.
      format(MSE_SKL, MSE_model))



#%% Visualization

trainAndModel = plt.figure()
feature = 6 # feature to visualate - col0 is an intercept column

plt.title('test data and model prediction')
plt.ylabel('target')
plt.xlabel(data.columns[feature-1]) 



#train data and model predictions
plt.scatter(X_test[:,feature],y_test,label='Test data', c='g', marker='o')


plt.scatter(X_test[:,feature], y_model, label='model prediction', c='b',
            marker = 'x')

linesX = np.ones((2,2))
linesX[0,1] = min(X_test[:,feature])
linesX[1,1] = max(X_test[:,feature])
#linesX = np.insert(linesX,0,np.ones(2), axis=1)
plt.plot(linesX[:,1], computePrediction(linesX,(theta[0],theta[feature])),
         label='model regresion line', c='b')

plt.legend()


# sklearn prediction vs model prediction
modelVsSKL = plt.figure()
plt.title('sklearn prediction vs model prediction')
plt.ylabel('target')
plt.xlabel(data.columns[feature-1]) 

#sklearn
plt.scatter(X_test[:,feature], y_SKL, label='sklearn prediction', c='r', 
            marker = '*', linewidths=2)
plt.plot(linesX[:,1], computePrediction(linesX,(thetaSKL[0],thetaSKL[feature])),
            label='skelarn regresion line', c='r', lw=5)
#model
plt.scatter(X_test[:,feature], y_model, label='model prediction', c='b',
            marker = 'x')
plt.plot(linesX[:,1], computePrediction(linesX,(theta[0],theta[feature])),
         label='model regresion line', c='b')

plt.legend()


"""
# more J visualization

par1 = 5 #theta values to visualate, par = 0 -> intercept coef
par2 = 6

thetaX = np.linspace(-30,30,100)
thetaY = np.linspace(-30,30,100)
Jxx, Jyy = np.meshgrid(thetaX, thetaY)
Z = np.ones(Jxx.shape)
thetaTemp = np.zeros(theta.shape)

for i in range(len(thetaX)):
    for j in range(len(thetaY)):
        thetaTemp[0] = thetaX[i]
        thetaTemp[1] = thetaY[j]
        Z[i,j] = computeCost(X, y, thetaTemp)

Z = np.transpose(Z)
#np.logspace(-0,3,num=15)
J_figure = plt.figure()
plt.contour(thetaX, thetaY, Z, np.logspace(0,3,num=30))
plt.scatter(theta[par1],theta[par2],marker='x', c='r', linewidths=50)
#plt.scatter(thetaSKL[0],thetaSKL[1],marker='*', c='k', linewidths=1)
plt.xlabel(data.columns[par1-1])
plt.ylabel(data.columns[par2-1])
plt.title('Cost as a function of theta')"""