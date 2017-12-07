# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:47:19 2017

@author: Michal

plotting ZIP codes map with clustering
"""
import pandas as pd
from mapToLatLon import PUWGToLatLonWGS84 as geo
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import numpy as np


#cleaning data
print('loading data')
df = pd.read_csv('lodzkie.csv', usecols=['kodPocztowy', 'pozycja/Point/pos'])
df = df.iloc[::100,:]
df = df[df.kodPocztowy != '00-000']
df = df.dropna(axis=0)

#converting to workable format
map_loc = list()
for row in range(len(df)):
    map_loc.append(df.iloc[row,1].split())
    
map_loc = pd.DataFrame(map_loc,columns=['X','Y'],dtype=float, index=df.index)

#converting to LON/LAT format
geo_loc = map_loc.apply(geo, axis=1)
geo_loc.columns = ['lat','lon']

#final results
df = df.drop('pozycja/Point/pos', axis=1)
df = pd.concat([df, geo_loc], axis=1)
#df.to_csv('lodzkie_geo.csv')
#%%
#df = pd.read_csv('lodzkie_geo.csv', index_col=0)

#removing outlayers
print('removing outlayers')
grouped_mean = df.groupby(['kodPocztowy']).mean()
grouped_std = df.groupby(['kodPocztowy']).std()
validator = pd.DataFrame(index=df.index)
validator['isCorrect'] = True
for row in df.index:
    kodPocztowy = df.loc[row,'kodPocztowy'] 
    if grouped_std.loc[kodPocztowy,'lat'] == 'nan':
        continue
    lat_max = grouped_mean.loc[kodPocztowy, 'lat'] + 2*grouped_std.loc[kodPocztowy, 'lat']
    lat_min = grouped_mean.loc[kodPocztowy, 'lat'] - 2*grouped_std.loc[kodPocztowy, 'lat']
    lon_max = grouped_mean.loc[kodPocztowy, 'lon'] + 2*grouped_std.loc[kodPocztowy, 'lon']
    lon_min = grouped_mean.loc[kodPocztowy, 'lon'] - 2*grouped_std.loc[kodPocztowy, 'lon']
   
    if (df.loc[row,'lat'] < lat_min) or(df.loc[row,'lat'] > lat_max) or (df.loc[row,'lon'] < lon_min) or (df.loc[row,'lon'] > lon_max):
        validator.loc[row,['isCorrect']] = False

df = df[validator.isCorrect]


#machine learning part
#%%
start_centers = df.drop_duplicates(subset='kodPocztowy')
start_centers = start_centers.drop(['kodPocztowy'], axis=1)

knn = MiniBatchKMeans(n_clusters=len(start_centers),init=start_centers,
                      max_iter=10)
print('learning')
knn.fit(df.loc[:,['lat','lon']])
#knn.fit(start_centers)


#%%
#presenting results
print('plotting')
plt.close('all')

cluster_centers = knn.cluster_centers_
plt.figure()
plt.scatter(cluster_centers[:,0],cluster_centers[:,1])
plt.title('start points')
plt.xlabel('lat')
plt.ylabel('lon')

#decision boundary
XX,YY = np.meshgrid(np.linspace(df.lat.min(),df.lat.max(),1000), 
                    np.linspace(df.lon.min(),df.lon.max(),1000))
mesh_conv = np.c_[XX.ravel(),YY.ravel()]
Z = knn.predict(mesh_conv)
Z = Z.reshape(XX.shape)

kontur_lodzkiego = np.flipud(np.load('kontur_lodzkiego.npy'))
final= np.multiply(Z,kontur_lodzkiego) 

plt.figure()
plt.title('filled countour')
plt.xlabel('lat')
plt.ylabel('lon')
plt.contourf(XX,YY,final, cmap=plt.cm.prism)

plt.figure()
plt.contour(XX,YY,final)
plt.title('contour')
plt.xlabel('lat')
plt.ylabel('lon')

print('end')

