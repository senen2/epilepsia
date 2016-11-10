'''
convolution on python
example of stanford

Created on Nov 4, 2016

@author: carlos
'''
import tensorflow as tf
import numpy as np
from apiepi import *

from matplotlib import pyplot as plt
import Image

def conv(x, c):
    return np.sum(x*c)

#im = plt.imread("taj_orig.png")
#im = np.array(Image.open("taj_orig.png"))

im = np.array([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]])
print im.shape
# plt.imshow(im)
# plt.show()
c = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # Edge detect
# c = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # Sharpen
#c = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) 
# c = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) # Edge detect
# c = np.array([[0, -.125, 0], [-.125, 1.5, -.125], [0, -.125, 0]]) # Edge detect
c = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
# print sum(sum(c))
# c = (c/4.0)*.1
new = np.zeros(im.shape)

#im = np.array([[1,2, 5],[3,4, 6],[13,14, 16]])
s = im.shape
im = np.insert(im, 0, 0, axis=0)
im = np.insert(im, 0, 0, axis=1)
im = np.insert(im, s[0]+1, 0, axis=0)
im = np.insert(im, s[1]+1, 0, axis=1)
print im.shape
# plt.imshow(im)
# plt.show()

xr = im.shape[0] - c.shape[0] + 1
yr = im.shape[1] - c.shape[1] + 1
cr = c.shape[0]

#for ch in xrange(3):
for i in xrange(xr):
    for j in xrange(yr):
        #a[0 :2, 0  :2]
        #a[begin :begin+length, begin :begin+length]
        #new[i, j, ch] = conv(im[i+1:i+1+cr, j+1:j+1+cr, ch], c) 
        #new[i-1, j-1, ch] = conv(im[i:i+cr, j:j+cr, ch], c) 
        new[i, j] = conv(im[i:i+cr, j:j+cr], c) #  test
        #new[i, j, ch] = conv(im[i:i+cr, j:j+cr, ch], c) #original 

# print new.shape
# plt.imshow(new)
# plt.show()
print new

