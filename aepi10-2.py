'''
Compute Normal probabilities and save file

Created on 14/09/2016
102589
@author: botpi
'''
import numpy as np
import scipy.io, scipy.stats
import os
import matplotlib.pyplot as plt

# Read samples
group = "train_1"
#group = "pp1"
directory = "c:/concursos/epilepsia/%s/" % group
directoryc = "c:/concursos/epilepsia/%sc/" % group
directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/epilepsia/%s/" % group

t = {}
for file in os.listdir(directory):
    try:
        print file
        mat = scipy.io.loadmat(directory + file)
        data = mat['dataStruct'][0][0][0]
        name = file.split(".")[0]

        b = np.sum(data, axis=1)
        c = data[b!=0]       

        #c = c[:, 1:3]
        
        t[name] = {}
        t[name]["mu"] = np.mean(c, axis=0) #, np.std(data)
        t[name]["sigma2"] = np.std(np.transpose(c), axis=1)**2
        t[name]["cov"] = np.cov(np.transpose(c))
        t[name]["corr"]= np.corrcoef(c, rowvar=0)
    except:
        pass

try:
    t.pop("__globals__")
    t.pop("__version__")
    t.pop("__header__")
except:
    pass

scipy.io.savemat(group + " no zeros", t, do_compression=True)
print "end"
