'''
Read mat file

Created on 08/09/2016

@author: botpi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle

mat = scipy.io.loadmat('c:/concursos/epilepsia/train_1/1_12_1.mat')
#mat = scipy.io.loadmat('c:/concursos/epilepsia/test_1/1_12.mat')
t = mat['dataStruct'][0][0]

print t.shape, len(t)
for a in t:
    print a.shape, len(a), len(a[0]), a[0]
#print mat

#print t[4][1]