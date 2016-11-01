'''
Read mean and standard deviation from pickle for types 1/0

Created on 12/09/2016

@author: botpi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import os

group = "train_1"
pickle_file = 'meandev_bytype_%s.pickle' % group

with open(pickle_file, 'rb') as file:
    try:
        while True:
            data = pickle.load(file)
    except EOFError:
        pass

t0 = data["0"]
t1 = data["1"]

print t0[100]["dev"]
print t1[100]["dev"]
