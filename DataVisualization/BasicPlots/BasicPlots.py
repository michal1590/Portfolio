# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:42:06 2017

@author: Michal

Basic plots: histograms, 2D and 3D plots
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

plt.style.use('ggplot')
plt.close('all')



#%%
#histogram

HistData = np.random.randn(500)

fig1 = plt.figure(1)
plt.hist(HistData,bins=10, rwidth=0.8)
plt.xlabel('values')
plt.ylabel('counts')
plt.title('histogram')



#%%
# 2D plot

x = np.linspace(0,10,100)
y = np.sin(x) + np.random.rand(100)

fig2 = plt.figure(2)
plt.plot(x,y,'blue')
plt.xlabel('x axis')
plt.ylabel('y values')
plt.title('2D plot')
plt.axis([-1, 11, 1.5*min(y), 1.5*max(y)])
plt.grid(True)



#%%
#ploting 3D

fig3 = plt.figure(3)

#line 3D
ax1 = fig3.add_subplot(121, projection='3d')

x3D = np.linspace(-5,5,50)
y3D = np.cos(x3D)
z3D = x3D*y3D
ax1.plot(x3D,y3D,z3D)
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')
plt.title('plot 3D')

#scatter points 3D
ax2 = fig3.add_subplot(122, projection='3d')

xs1 = np.random.rand(10)
ys1 = np.random.rand(10)
zs1 = np.random.rand(10)
ax2.scatter(xs1,ys1,zs1, c='r', marker='o', label='rand')

xs2 = np.random.randn(10)
ys2 = np.random.randn(10)
zs2 = np.random.randn(10)
ax2.scatter(xs2,ys2,zs2, c='g', marker='*', label='randn')

plt.legend()
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
ax2.set_zlabel('z axis')
plt.title('scatter 3D')


#surface 3D
fig4 = plt.figure(4)

x = np.linspace(-2*np.pi,2*np.pi,100)
y = np.linspace(-2*np.pi,2*np.pi,100)

X,Y = np.meshgrid(x,y)
Z = np.sin(X)+np.cos(Y)

ax3 = fig4.add_subplot(111, projection='3d')
ax3.plot_surface(X,Y,Z,cmap='cool' )
plt.title('surface 3D')
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')



plt.show()


