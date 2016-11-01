'''
Calculates mean and standard deviation and save for each type 1/0

Created on 12/09/2016

@author: botpi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import os

def meandev(type, directory):
    t = []
    for file in os.listdir(directory):
        if file.endswith(type + ".mat"):
            print file
            r = {}
            r["file"] = file
            mat = scipy.io.loadmat(directory + file)
            data = mat['dataStruct'][0][0][0] 
            r["mean"] = np.mean(np.transpose(data), axis=1) #, np.std(data)
            r["dev"] = np.std(np.transpose(data), axis=1) 
            t.append(r)
    return t


group = "train_1"
directory = "c:/concursos/epilepsia/%s/" % group

t0 = meandev("0", directory)
t1 = meandev("1", directory)

pickle_file = 'meandev_bytype_%s.pickle' % group

try:
  f = open(pickle_file, 'wb')
  save = {
    '0': t0,
    '1': t1,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
