'''
Plot histrogram of channel

Created on 12/09/2016

@author: botpi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

directory = "train_1"
file = "1_12_1"
channel = 13

mat = scipy.io.loadmat('c:/concursos/epilepsia/%s/%s.mat' % (directory, file))
t = np.transpose(mat['dataStruct'][0][0][0])

# print t[1]
# print len(t)

plt.hist(t[channel], bins=100)
#plt.hist(t[0])
plt.title("Histogram")
fig = plt.gcf()
fig.canvas.set_window_title(file)
plt.show()
#plt.savefig("pp.png")