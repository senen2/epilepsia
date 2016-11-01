'''
Plot histrogram of every channel for a file

Created on 12/09/2016

@author: botpi
'''
import numpy as np
import scipy.io
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

directory = "train_1"
file = "1_953_0"
channel = 13

mat = scipy.io.loadmat('c:/concursos/epilepsia/%s/%s.mat' % (directory, file))
t = np.transpose(mat['dataStruct'][0][0][0])


for channel in range(16):
	plt.hist(t[channel], bins=100)

	plt.title("Histogram")
	fig = plt.gcf()
	fig.canvas.set_window_title("%s-%s" % (file, channel) )
	#plt.show()
	plt.savefig("%s-%s.png" % (file, channel))