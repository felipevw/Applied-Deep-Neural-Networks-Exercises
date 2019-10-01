
"""
# Feedforward Neural Networks Part 1: understanding overfitting
Created on Wed Sep 25 10:36:37 2019

@author: felip
"""


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.optimize import curve_fit

# Real model 
def func_2(p, a, b, c):
    return a + b * p + c * p **2

# Generate the dataset
x = np.arange(-5.0, 5.0, 0.05, dtype = np.float64)
y = func_2(x, 1, 2, 3) + 18.0 * np.random.normal(0, 1, size = len(x))

# Possible models of the dataset
def func_1(p, a, b):
    return a + b * p

def func_14(p, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
    return a+b*p + c*p**2+d*p**3+e*p**4 + f*p**5 + g*p**6 + h*p**7+i*p**8 + j*p**9+k*p**10+l*p**11 + m*p**12 + n*p**13 + o*p**14


# Curve fitting
popt1, pcov1 = curve_fit(func_1, x, y)
popt2, pcov2 = curve_fit(func_2, x, y)
popt14, pcov14 = curve_fit(func_14, x, y)


# Plot the dataset
plt.rc('font', family='arial')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
    
plt.tight_layout()

fig1 = plt.figure(figsize=(10,8))
ax = fig1.add_subplot(1, 1, 1)
ax.scatter(x, y, ls='solid', color = 'red')
ax.plot(x, func_1(x, popt1[0], popt1[1]), ls='solid', color = 'blue')

ax.set_title('Model of the dataset for a 1-degree polynomial', fontsize = 16)
ax.set_xlabel('X values', fontsize = 16)
ax.set_ylabel('Y values', fontsize = 16)


fig2 = plt.figure(figsize=(10,8))
ay = fig2.add_subplot(1, 1, 1)
ay.scatter(x, y, ls='solid', color = 'red')
ay.plot(x, func_2(x, popt2[0], popt2[1], popt2[2]), ls='solid', color = 'blue')

ay.set_title('Model of the dataset for a 2-degree polynomial', fontsize = 16)
ay.set_xlabel('X values', fontsize = 16)
ay.set_ylabel('Y values', fontsize = 16)


fig2 = plt.figure(figsize=(10,8))
az = fig2.add_subplot(1, 1, 1)
az.scatter(x, y, ls='solid', color = 'red')
az.plot(x, func_14(x, *popt14), ls='solid', color = 'blue')

az.set_title('Model of the dataset for a 14-degree polynomial', fontsize = 16)
az.set_xlabel('X values', fontsize = 16)
az.set_ylabel('Y values', fontsize = 16)



