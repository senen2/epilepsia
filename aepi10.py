'''
Compute Normal probabilities and save file

Created on 14/09/2016

@author: botpi
'''
import numpy as np
import scipy.io, scipy.stats
#from scipy.stats import multivariate_normal
import os
import matplotlib.pyplot as plt

# Read samples
group = "train_1"
group = "pp1"
directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/epilepsia/%s/" % group
t = {}
for file in os.listdir(directory):
#     try:
        print file
        mat = scipy.io.loadmat(directory + file)
        if "dataStruct" in mat:
            data = mat['dataStruct'][0][0][0]
        else:
            data = mat["c"]
        name = file.split(".")[0]
        t[name] = {}
        mu = np.mean(np.transpose(data), axis=1) #, np.std(data)
        sigma2 = np.std(np.transpose(data), axis=1)**2
        t[name] = scipy.stats.multivariate_normal.pdf(data, mu, sigma2)
        #plt.hist(data, bins=100)
#         print t[name].shape
#         plt.hist(t[name], bins=100)
#         fig = plt.gcf()
#         fig.canvas.set_window_title(name)
#         plt.show()
        print name, mu, sigma2
#         print name, t[name].shape, t[name]
#     except:
#         pass

try:
    t.pop("__globals__")
    t.pop("__version__")
    t.pop("__header__")
except:
    pass

scipy.io.savemat(group + "_a", t, do_compression=True)
print "end"
