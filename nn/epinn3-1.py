'''
Create submission

Created on Oct 21, 2016
begin
1584 1584
positivos 612 totales 1584
2256 2256
positivos 31 totales 2256
2286 2286
positivos 30 totales 2286
gran total 6127
end
@author: botpi
'''
import tensorflow as tf
import numpy as np
import scipy.io
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
    #print resp["W1"].shape, resp["b1"][0].shape, resp["W2"].shape, resp["b2"][0].shape, images.shape, id
    #print np.isnan(resp["W1"]).any(), np.isnan(resp["b1"][0]).any(), np.isnan(resp["W2"]).any(), np.isnan(resp["b2"][0]).any()
    
#     ones = np.ones([16,16])
#     z = np.diag(np.ones(16))
#     for im in images:
#         if np.sum(im==z)==256:
#             im = ones
    
    with tf.Session() as sess:
        sess.run(init)
        prob = pred.eval({x:images, W1:resp["W1"], b1:resp["b1"][0], W2:resp["W2"], b2:resp["b2"][0] })
        #probm = predm.eval({x:images, W:resp["W"], b:resp["b"] })
    print len(prob), len(names)
    
    p = 0
    for i in xrange(len(names)):

#         if names[i] == "1_1120": # 1_1027
#             a = prob[i]
#         if np.isnan(prob[i][1]):
#             prob[i][1] = 0        
        r.append([names[i] + ".mat", prob[i][1]])
        if prob[i][0] < prob[i][1]:
            p += 1
    print "positivos", p, "totales", len(names)

print "gran total", len(r)
np.savetxt("submission_nn_7.csv", r, delimiter=',', fmt="%s,%s")

print "end"