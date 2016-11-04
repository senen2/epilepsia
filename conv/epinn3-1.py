'''
Create submission

Created on Oct 21, 2016

block best solution
positivos 565 totales 1584
positivos 31 totales 2256
positivos 30 totales 2286
gran total 6127

0 and 1
positivos 726 totales 1584
positivos 3 totales 2256
positivos 17 totales 2286
gran total 6127

@author: botpi
'''
import tensorflow as tf
import numpy as np
from apiepi import *

print "begin"

hidden = 10
W1 = tf.Variable(tf.zeros([256,hidden]))
b1 = tf.Variable(tf.zeros([hidden]))
W2 = tf.Variable(tf.zeros([hidden,2]))
b2 = tf.Variable(tf.zeros([2]))

x = tf.placeholder(tf.float32, [None, 256])
x1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
pred = tf.nn.softmax(tf.matmul(x1, W2) + b2)

init = tf.initialize_all_variables()

r = []
r.append(["File", "Class"])

for i in range(3):
    id = i+1
    resp = scipy.io.loadmat("resp_%s" % id)
    images, labels, names = read_images("test_%s nz" % id)
    
    with tf.Session() as sess:
        sess.run(init)
        prob = pred.eval({x:images, W1:resp["W1"], b1:resp["b1"][0], W2:resp["W2"], b2:resp["b2"][0] })
    
    p = 0
    for i in xrange(len(names)):
        r.append([names[i] + ".mat", prob[i][1]])
        if prob[i][0] < prob[i][1]:
            p += 1
    print "positivos", p, "totales", len(names)

print "gran total", len(r)
np.savetxt("submission_nn_10.csv", r, delimiter=',', fmt="%s,%s")

print "end"