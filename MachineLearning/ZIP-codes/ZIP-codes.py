# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:07:42 2017

@author: Michal

ZIP codes clustering
"""
"""
import pandas as pd
import math as m

def CalculateESquared ( a,  b):
    return ((a * a) - (b * b)) / (a * a)


def CalculateE2Squared (a,b):
    return ((a * a) - (b * b)) / (b * b)


def denom (es, sphi):
    sinSphi = m.sin (sphi)
    return m.sqrt (1.0 - es * (sinSphi * sinSphi));


def sphsr (a,es,sphi):
    dn = denom (es, sphi)
    return a * (1.0 - es) / (dn * dn * dn)

def sphsn (a, es, sphi):
    sinSphi = m.sin (sphi)
    return a / m.sqrt (1.0 - es * (sinSphi * sinSphi))


def sphtmd (ap,bp, cp, dp, ep, sphi):
    return (ap * sphi) - (bp * m.sin (2.0 * sphi)) + (cp * m.sin (4.0 * sphi))- (dp * m.sin (6.0 * sphi)) + (ep * m.sin (8.0 * sphi))




def PUWGToLatLon (a, f, easting, northing):

    fe = 500000.0
    ok =  0.9993
    recf = 1.0 / f
    b = a * (recf - 1) / recf
    eSquared = CalculateESquared (a, b)
    e2Squared = CalculateE2Squared (a, b)
    tn = (a - b) / (a + b);
    ap = a * (1.0 - tn + 5.0 * ((tn * tn) - (tn * tn * tn)) / 4.0 + 81.0 * ((tn * tn * tn * tn) - (tn * tn * tn * tn * tn)) / 64.0)
    bp = 3.0 * a * (tn - (tn * tn) + 7.0 * ((tn * tn * tn) - (tn * tn * tn * tn)) / 8.0 + 55.0 * (tn * tn * tn * tn * tn) / 64.0) / 2.0
    cp = 15.0 * a * ((tn * tn) - (tn * tn * tn) + 3.0 * ((tn * tn * tn * tn) - (tn * tn * tn * tn * tn)) / 4.0) / 16.0
    dp = 35.0 * a * ((tn * tn * tn) - (tn * tn * tn * tn) + 11.0 * (tn * tn * tn * tn * tn) / 16.0) / 48.0;
    ep = 315.0 * a * ((tn * tn * tn * tn) - (tn * tn * tn * tn * tn)) / 512.0
    olam = m.radians(19.0) 
    strf = 0.0
    nfn = -5300000.0
    tmd = (northing - nfn) / ok
    sr = sphsr (a, eSquared, 0.0)
    ftphi = tmd / sr
#    double t10, t11, t12, t13, t14, t15, t16, t17
    for i in range(5):
        t10 = sphtmd (ap, bp, cp, dp, ep, ftphi)
        sr = sphsr (a, eSquared, ftphi)
        ftphi = ftphi + (tmd - t10) / sr
    
    sr = sphsr (a, eSquared, ftphi)
    sn = sphsn (a, eSquared, ftphi)
    s = m.sin (ftphi)
    c = m.cos (ftphi)
    t = s / c
    eta = e2Squared * (c * c)
    de = easting - fe - strf
    t10 = t / (2.0 * sr * sn * (ok * ok))
    t11 = t * (5.0 + 3.0 * (t * t) + eta - 4.0 * (eta * eta) - 9.0 * (t * t) * eta) / (24.0 * sr * (sn * sn * sn) * (ok * ok * ok * ok))
    t12 = t *  (61.0 + 90.0 * (t*t) + 46.0 * eta + 45.0 * (t* t * t * t) - 252.0 * (t * t) * eta - 3.0 * (eta * eta) + 100.0 * (eta * eta * eta) - 66.0 * (t * t) * (eta * eta) - 90.0 * (t * t * t * t) * eta + 88.0 * (eta * eta * eta * eta) + 225.0 * (t * t * t * t) * (eta * eta) + 84.0 * (t * t) * (eta * eta * eta) - 192.0 * (t * t) * (eta * eta * eta * eta)) / (720.0 * sr * (sn * sn * sn* sn * sn ) * (ok * ok * ok * ok * ok * ok))
    t13 = t * (1385.0 + 3633 * (t * t) + 4095.0 * (t * t * t * t) + 1575.0  * (t * t * t * t * t *t)) / (40320 * sr * (sn * sn * sn* sn * sn * sn * sn ) * (ok * ok * ok * ok * ok * ok * ok * ok))
    lat = ftphi - (de * de) * t10 + (de * de * de * de) * t11 - (de * de * de * de * de * de) * t12 + (de * de * de * de * de * de * de * de) * t13
    t14 = 1.0 / (sn * c * ok)
    t15 = (1.0 + 2.0 * (t * t) + eta) / (6.0 * (sn * sn * sn) * c * (ok * ok * ok))
    t16 = 1.0 * (5.0 + 6.0 * eta + 28.0 * (t * t) - 3.0 * (eta * eta) + 8.0 * (t * t) * eta + 24.0 * (t * t * t * t) - 4.0 * (eta * eta * eta) + 4.0 *(t * t) * (eta * eta) + 24.0 * (t * t) * (eta * eta * eta)) / (120.0 * (sn * sn * sn * sn * sn) * c * (ok * ok * ok * ok * ok))
    t17 = 1.0 * (61.0 + 662.0 * (t * t) + 1320.0 * (t * t * t * t) + 720.0 * (t * t * t * t * t * t)) / (5040.0 * (sn * sn * sn * sn * sn * sn * sn) * c * (ok * ok * ok * ok * ok * ok * ok))
    dlam = de * t14 - (de * de * de) * t15 + (de * de * de * de * de) * t16 - (de * de * de * de * de * de * de) * t17
    lon = olam + dlam
    lon =m.degrees(lon)
    lat = m.degrees(lat)
    return lat, lon
    
#%%    
def PUWGToLatLonWGS84(x):
    
    easting = x[0]
    northing = x[1]
    
    (lat,lon) = PUWGToLatLon (6378137.0, 1 / 298.257223563, easting, northing)
    return lat, lon
    



#%%
#data = pd.read_csv('dane.csv')
#dolnyslask = pd.read_csv('dolnoslaskie.csv')
lodzkie = pd.read_csv('lodzkie.csv')
lodzkie.drop(['%Key_FeatureCollection_BC87E8165946E7D6',
       'cyklZycia/BT_CyklZyciaInfo/poczatekWersjiObiektu', 'czescMiejscowosci',
       'czescMiejscowosci/nilReason', 'gml:id',
       'identifier', 'identifier/codeSpace',
       'idIIP/BT_Identyfikator/lokalnyId',
       'idIIP/BT_Identyfikator/przestrzenNazw',
       'idIIP/BT_Identyfikator/wersjaId', 'obiektEMUiA/xlink:href', 'pozycja/Point/gml:id',
       'pozycja/Point/srsName', 'status', 'ulica/nilReason',
       'ulica/xsi:nil', 'waznyDo', 'waznyOd', 'waznyOd/nilReason',
       'waznyOd/xsi:nil'], axis=1, inplace=True)
#kujpom = pd.read_csv('kujpom.csv')
#%%
wspolrzedne_plaskie = list(())

#rozbijam pozycje na dwie osobne kolumny    
for wiersz in range(lodzkie.shape[0]):
    wspolrzedne_plaskie.append(lodzkie.loc[wiersz,'pozycja/Point/pos'].split())

#tworze df z konwersja na float
wspolrzedne_plaskie = pd.DataFrame(wspolrzedne_plaskie,columns=['X','Y'],dtype=float)
    
#(lat,lon) = PUWGToLatLonWGS84(wspolrzedne_plaskie.iloc[1,:])

test = wspolrzedne_plaskie.iloc[:,:]
# obliczam wspol geo dla kazdego wiersza
wynik = test.apply(PUWGToLatLonWGS84, axis=1)

#rozbijam wynik na dwie kolumny
wspolrzedne_geo = list(())
for wiersz in range(len(wynik)):
    wspolrzedne_geo.append(wynik.iloc[wiersz])

wspolrzedne_geo = pd.DataFrame(wspolrzedne_geo, columns=['lat','lon'])

final = pd.concat([lodzkie, wspolrzedne_plaskie,wspolrzedne_geo], axis=1)
final.to_csv('muggled.csv')

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
knn = cluster.MiniBatchKMeans(n_clusters=start_centers.shape[0],init=start_centers, max_iter=1, n_init=1)
#knn = cluster.MiniBatchKMeans(n_clusters=grouped_mean.shape[0],init=grouped_mean)
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