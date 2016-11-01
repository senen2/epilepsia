'''
Created on Oct 21, 2016

@author: botpi
'''
import numpy as np
import scipy.io
from apiepi import *
from apiepi import sigmoid

print "begin"
group = "resp_1-1"
resp = scipy.io.loadmat(group)
group = "train_1"
images, labels = read_images(group)

# print resp["W"], resp["b"]
# print resp["W"].shape, resp["b"].shape

x1 = sigmoid(images.dot(resp["W1"]) + resp["b1"])
pred = sigmoid(x1.dot(resp["W2"]) + resp["b2"])

#print pred, resp["b"]
print np.sum(np.argmax(pred, 1)), np.sum(np.argmax(labels, 1))
correct_prediction = np.equal(np.argmax(pred, 1), np.argmax(labels, 1))
print np.sum(correct_prediction), np.sum(correct_prediction) / labels.shape[0]
accuracy = np.mean((correct_prediction))
print accuracy
print 1128/1267.
print "end"