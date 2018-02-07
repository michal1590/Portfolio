# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:31:22 2017

@author: Michal

function to transfor coordinates from Państwowy Układ Współrzędnych Geodezyjnych 1992
to LON/LAT system


Based on http://www.szymanski-net.eu/programy.html, Autor: Zbigniew Szymanski

input data should be in (X,Y) form

"""
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
    
   
def PUWGToLatLonWGS84(x):
    
    easting = x[0]
    northing = x[1]
    
    (lat,lon) = PUWGToLatLon (6378137.0, 1 / 298.257223563, easting, northing)
    return [lat, lon]
    
