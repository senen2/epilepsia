'''
Get file of max deviation

Created on 12/09/2016

@author: botpi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import os
import matplotlib.pyplot as plt

group = "train_1"
pickle_file = 'meandev_byfile_%s.pickle' % group
file = "1_12_1"
file = "1_1002_0"

with open(pickle_file, 'rb') as f:
    try:
        while True:
            data = pickle.load(f)
    except EOFError:
        pass

dev0 = []
dev1 = []
for t in data:
    if t[-1] == "0":
        dev0.append(data[t]["dev"])
    else:
        dev1.append(data[t]["dev"])
    
dev0 = np.transpose(np.array(dev0))
dev1 = np.transpose(np.array(dev1))    

min0 = np.min(dev0, axis=1)
min1 = np.min(dev1, axis=1)
max0 = np.max(dev0, axis=1)
max1 = np.max(dev1, axis=1)
mean0 = np.mean(dev0, axis=1)
mean1 = np.mean(dev1, axis=1)

for i in range(16):
    for t in data:
        if t[-1] == "0":
        	for i in range(16):
	            if data[t]["dev"][i] == max0[i]:
	    	    	t0 = t
	    	    	break

print t, t0, i
#print dev0.shape, np.min(dev0, axis=1), max0, mean0
#print dev1.shape, np.min(dev1, axis=1), max1, mean1
        
# plt.hist(data[file]["dev"], bins=16)
# fig = plt.gcf()
# fig.canvas.set_window_title(type)
# plt.show()