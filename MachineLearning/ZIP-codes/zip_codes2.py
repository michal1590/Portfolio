# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:16:57 2017

@author: Michal
"""

#%% loading data
import pandas as pd
import numpy as np

df = pd.read_csv('dlugi.csv', index_col=0, encoding='utf-16', na_values='nan',
                 dtype={'kodPocztowy':'str', 'lon':np.float64,'lat':np.float64})

df = df.iloc[::10,[2,10,11]]
df = df[df.kodPocztowy != '00-000']
df = df[df.kodPocztowy != 'nan']

df.dropna(axis=0, inplace=True)
print(df.isnull().any())
#%%
grouped_mean = df.groupby(['kodPocztowy']).mean()
grouped_std = df.groupby(['kodPocztowy']).std()
#size = df.shape



#unique = df.loc[:,'kodPocztowy'].unique().
start_centers = df.drop_duplicates(subset='kodPocztowy')
start_centers.lat = pd.to_numeric(start_centers.lat)
start_centers.lon = pd.to_numeric(start_centers.lon)
start_centers = start_centers.drop(['kodPocztowy'], axis=1)

from sklearn import cluster
print('ucze sie')
#knn = cluster.MiniBatchKMeans(n_clusters=start_centers.shape[0],init=start_centers.values, max_iter=1, n_init=1)
knn = cluster.MiniBatchKMeans(n_clusters=grouped_mean.shape[0],init=grouped_mean.values, max_iter=1, n_init=1)
test_data = df.loc[:,['lat','lon']]
knn.fit(test_data)

print('ma;uje')
import matplotlib.pyplot as plt
plt.close('all')
cluster_centers = knn.cluster_centers_
plt.figure()
plt.scatter(cluster_centers[:,0],cluster_centers[:,1])
#%%
#decision boundary
XX,YY = np.meshgrid(np.linspace(df.lat.min(),df.lat.max(),500), np.linspace(df.lon.min(),df.lon.max(),500))
mesh_conv = np.c_[XX.ravel(),YY.ravel()]
Z = knn.predict(mesh_conv)
Z = Z.reshape(XX.shape)

plt.figure()
plt.contourf(XX,YY,Z, cmap=plt.cm.prism)
plt.figure()
plt.contour(XX,YY,Z)
print('koniec')
#stworzyc zip-code -> mean(wszystkie zipy)
# usuwamy te odstajace
#usuwamy te z 00-00 und NA