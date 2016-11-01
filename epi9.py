'''
Calculate product of probabilites

Created on 14/09/2016

@author: papi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import os
import matplotlib.pyplot as plt

K1 = 1 / np.sqrt(2*np.pi)

def gauss(x, mean, dev):
    return np.exp(-(x-mean)*(x-mean)/2/dev*dev) / K1 / dev
    

group = "train_1"
pickle_file = 'meandev_byfile_%s.pickle' % group
directory = "c:/concursos/epilepsia/%s/" % group

with open(pickle_file, 'rb') as f:
    meandev = pickle.load(f)

for file in meandev:
    mat = scipy.io.loadmat(directory + file)
    data = mat['dataStruct'][0][0][0]
    print file, data[0]
    print meandev[file]["mean"], meandev[file]["dev"]
    print gauss(data[0], meandev[file]["mean"], meandev[file]["dev"])
    #a()