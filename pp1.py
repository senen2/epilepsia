'''
Created on 12/09/2016

@author: papi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import os
import matplotlib.pyplot as plt

a = np.array([1,2,3,4,5,6,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4])
plt.hist(a, bins=3)
plt.show()

