'''
calculates mean and standard deviation and save for each file

Created on 12/09/2016

@author: botpi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import os

def meandev(directory):
    t = {}
    for file in os.listdir(directory):
        try:
            print file
            mat = scipy.io.loadmat(directory + file)
            data = mat['dataStruct'][0][0][0]
            name = file.split(".")[0]
            t[name] = {}
            t[name]["mean"] = np.mean(np.transpose(data), axis=1) #, np.std(data)
            t[name]["dev"] = np.std(np.transpose(data), axis=1)
            print t[name]["mean"]
        except:
            pass    
    return t

group = "train_1"
directory = "c:/concursos/epilepsia/%s/" % group

t = meandev(directory)

pickle_file = 'meandev_byfile_%s.pickle' % group

with open(pickle_file, 'wb') as f:
    pickle.dump(t, f)
    f.close()
