'''
Evaluate predicion by dev

Created on 14/09/2016

@author: papi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import os
import matplotlib.pyplot as plt

group = "train_1"
pickle_file = 'meandev_byfile_%s.pickle' % group

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

rp = 0
rn = 0
p = 0
n = 0
fp = 0
fn = 0

for f in data:
    #print data[f]["dev"][0]
    if data[f]["dev"][0] < 20:
        r = 1
    else:
        r = 0
  
    if f[-1:] == "1":
        rp += 1
    else:
        rn += 1
    
    if f[-1:] == str(r):
        if r == 1:
            p += 1
        else:
            n += 1
    elif f[-1:] == "0" and r == 1:
        fp += 1
    else:
        fn += 1
        
print "total = ", len(data)
print "rp = ", rp
print "rn = ", rn
print
print "p = ", p
print "n = ", n
print "fp = ", fp
print "fn = ", fn
        
         