'''
Compute correlation coefficient and save file

Created on 14/09/2016
102589

test 1
total 1584, nan 47, not nan 1537

test 2
total 2256 nan 29

test 3
total 2286 nan 4

@author: botpi
'''
import numpy as np
import scipy.io, scipy.stats
import os
import matplotlib.pyplot as plt

print "begin"
# Read samples
group = "test_3"
#group = "pp1"
directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/epilepsia/%s/" % group
n = 0
#z = np.diag(np.ones(16))
z = np.ones([16,16])
t = {}
for file in os.listdir(directory):
    try:
        #print file
        mat = scipy.io.loadmat(directory + file)
        data = mat['dataStruct'][0][0][0]
        name = file.split(".")[0]

        b = np.sum(data, axis=1)
        c = data[b!=0]       

        #c = c[:, 1:3]
        
        t[name] = {}
#         t[name]["mu"] = np.mean(c, axis=0) #, np.std(data)
#         t[name]["sigma2"] = np.std(np.transpose(c), axis=1)**2
#         t[name]["cov"] = np.cov(np.transpose(c))
        t[name]["corr"] = np.corrcoef(c, rowvar=0)
        if np.isnan(t[name]["corr"]).any():
            print name
            t[name]["corr"] = z
            n+=1
    except:
        pass

try:
    t.pop("__globals__")
    t.pop("__version__")
    t.pop("__header__")
except:
    pass

print len(t), n, len(t)+n
scipy.io.savemat(group + " nz", t, do_compression=True)
print "end"
