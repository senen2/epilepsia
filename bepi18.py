'''
Read data and plot

@author: botpi
'''
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

print "reading..."

group = "train_1"
group = "pp1"
t = scipy.io.loadmat(group + "f")
t.pop("__globals__")
t.pop("__version__")
t.pop("__header__")

print t["1_2_0"].shape
plt.hist(t["1_2_0"], bins=100)
plt.show()