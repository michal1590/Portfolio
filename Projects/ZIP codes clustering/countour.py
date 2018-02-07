# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:39:12 2017

@author: Michal

creating contour of area
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

img_grayscale = cv2.imread('woj_5.png', cv2.IMREAD_GRAYSCALE) # loading to grayscale 

_,img_bin = cv2.threshold(img_grayscale,245,255,cv2.THRESH_BINARY_INV) #producing binary image
median = cv2.medianBlur(img_bin,5) #removing noise with median filter

# removing rows and cols without desired field
sum_cols = median.sum(axis=0)
img_reduced = median[:,sum_cols!=0]

sum_rows = median.sum(axis=1)
img_reduced = img_reduced[sum_rows!=0,:]


#resize
resized = cv2.resize(img_reduced,(1000,1000), interpolation = cv2.INTER_CUBIC) #result needs to be binarized again
_,img_bin = cv2.threshold(resized,100,255,cv2.THRESH_BINARY) #producing binary image
median = cv2.medianBlur(img_bin,7) #removing noise with median filter
img = (median/255).astype(np.uint8)


plt.imshow(img)
