# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:43:48 2017

@author: Michal

Plots with Pandas
"""
import pandas as pd
from sklearn.datasets import  load_iris
import matplotlib.pyplot as plt

plt.close('all')

#preparing data
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.Series(data.target)
target = target.map({0:data.target_names[0],1:data.target_names[1],
            2:data.target_names[2]})
df['target'] = target


#bar plot to visualize differencies among species
df.groupby('target').mean().plot.bar(title='bar plot')

#more insights
df.plot.box(by='target', title='box plot')

#Andrews curve
plt.figure()
pd.plotting.andrews_curves(df,'target')

#Parallel coordinates - similar chart to previous one
plt.figure()
pd.plotting.parallel_coordinates(df,'target')
