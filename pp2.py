'''
Compute Normal probabilities

Created on 14/09/2016

@author: botpi
'''
import numpy as np
import scipy.io, scipy.stats
from six.moves import cPickle as pickle
import os

# Read samples
group = "train_1"
directory = "c:/concursos/epilepsia/%s/" % group
t = {}
for file in os.listdir(directory):
    print file
    mat = scipy.io.loadmat(directory + file)
    #data = mat['dataStruct'][0][0][0]
    
    scipy.io.savemat("pp2", mat, do_compression=True)
    a()
