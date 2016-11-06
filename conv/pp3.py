'''
Envia y recibe una imagen a tensorflow
Created on Nov 4, 2016

@author: carlos
'''
import tensorflow as tf
import numpy as np
from apiepi import *

from matplotlib import pyplot as plt

def conv(x, c):
    return np.sum(x*c)

im = plt.imread("taj_orig.png")
print im.shape
c = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # Edge detect
#c = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # Sharpen
#c = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) 

new = np.zeros(im.shape)
xr = im.shape[0] - c.shape[0] + 1
yr = im.shape[1] - c.shape[1] + 1
cr = c.shape[0]

for ch in xrange(cr):
    for i in xrange(xr):
        for j in xrange(yr):
            #a[0 :2 ,0  :2 ]
            new[i, j, ch] = conv(im[i:i+cr, j:j+cr, ch], c) 

plt.imshow(new/9)
plt.show()

