'''
Plot probabilites

Created on 14/09/2016

@author: papi
'''
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

group = "train_1"
group = "pp1"
t = scipy.io.loadmat(group)
t.pop("__globals__")
t.pop("__version__")
t.pop("__header__")

bestEpsilon = 0
bestF1 = 0

for d in t:
    print d
    a = t[d][0]#.reshape(-1, 100).mean(axis=1)
    print a.shape
    fig = plt.gcf()
    fig.canvas.set_window_title(d)
    plt.hist(a, bins=np.arange(0,1e-30,.5e-32))
    #plt.plot(a)
    plt.show()
