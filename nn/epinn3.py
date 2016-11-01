'''
Create submission

Created on Oct 21, 2016

@author: botpi
'''
import tensorflow as tf
import numpy as np
import scipy.io
from apiepi import *

print "begin"

x = tf.placeholder(tf.float32, [None, 256])
W = tf.placeholder(tf.float32, [256, 2])
b = tf.placeholder(tf.float32, [1, 2])
pred = tf.nn.softmax(tf.matmul(x, W) + b)
predm = tf.arg_max(pred, 1)
init = tf.initialize_all_variables()

r = []
r.append(["File", "Class"])

for i in range(3):
    id = i+1
    resp = scipy.io.loadmat("resp_%s" % id)
    images, labels, names = read_images("test_%s" % id)
    
    with tf.Session() as sess:
        sess.run(init)
        prob = pred.eval({x:images, W:resp["W"], b:resp["b"] })
        probm = predm.eval({x:images, W:resp["W"], b:resp["b"] })

    print "AUC", auc(labels, prob)
    
    p = 0
    for i in xrange(len(names)):
#         if tf.is_nan(prob[i][1]):
#             prob[i][1] = 0
#         if names[i] == "1_523":
#             a = prob[i]
#             pass
        r.append([names[i] + ".mat", prob[i][1]])
#         r.append([names[i] + ".mat", probm[i]])
#         if probm[i][0] < prob[i][1]:
#             p += 1
    print "positivos", p, "totales", len(names)

np.savetxt("submission_nn_4.csv", r, delimiter=',', fmt="%s,%s")

print "end"