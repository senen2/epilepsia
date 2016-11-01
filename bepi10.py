'''
Compute Normal probabilities and save file

Created on 14/09/2016

train_1 17 seg, 1301
train_2 21 seg, 2346
train_3 19 seg, 2394

train_1 19 seg, 1584
train_2 21 seg, 2256
train_3 19 seg, 2286

@author: botpi
'''
import numpy as np
import scipy.io, scipy.stats
import os
import matplotlib.pyplot as plt
import kronos

k = kronos.krono()
# Read samples
group = "test_1"
#group = "pp1"
directory = "c:/concursos/epilepsia/%s/" % group
t = {}
for file in os.listdir(directory):
    try:
        #print file
        mat = scipy.io.loadmat(directory + file)
        if "dataStruct" in mat:
            data = mat['dataStruct'][0][0][0]
        else:
            data = mat["c"]
        name = file.split(".")[0]
        t[name] = data
    except:
        pass

try:
    t.pop("__globals__")
    t.pop("__version__")
    t.pop("__header__")
except:
    pass

#scipy.io.savemat(group + "f", t, do_compression=True)
print k.elapsed(), "seg"
print "end"
