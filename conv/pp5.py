'''
Envia y recibe una imagen a tensorflow
Created on Nov 4, 2016

@author: carlos
'''
import tensorflow as tf
import numpy as np
from apiepi import *

from matplotlib import pyplot as plt
from scipy import signal as sg

im = plt.imread("taj_orig.png")
c = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # Edge detect
# c = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # Sharpen
#c = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) 

new = sg.convolve2d(im, c, "full")

plt.imshow(new/6)
plt.show()

