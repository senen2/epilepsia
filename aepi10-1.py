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
group = "pp1"
directory = "c:/concursos/epilepsia/%s/" % group
directoryc = "c:/concursos/epilepsia/%sc/" % group

t = {}
for file in os.listdir(directory):
    try:
        print file
        mat = scipy.io.loadmat(directory + file)
        data = mat['dataStruct'][0][0][0]
        name = file.split(".")[0]

        b = np.sum(data, axis=1)
        c = data[b!=0]
        
        a = {}
        a["c"] = c
        scipy.io.savemat(directoryc + name, a, do_compression=True)
    except:
        pass
print "end"
