'''
Calculates mean and dev for a file

Created on 09/09/2016

@author: botpi
'''
import numpy as np
import scipy.io

mat = scipy.io.loadmat('c:/concursos/epilepsia/train_1/1_12_1.mat')
data = mat['dataStruct'][0][0][0]

mean = np.mean(np.transpose(data), axis=1) #, np.std(data)
dev = np.std(np.transpose(data), axis=1) 
print mean, len(mean)
print dev, len(dev)