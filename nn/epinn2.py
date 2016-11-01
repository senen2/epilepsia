'''
Calculate W, b and save

Created on Oct 21, 2016

Train_1
Accuracy: 0.890292 epochs 1000 learning rate 5e-05
AUC 0.776613005451

Train_2
Accuracy: 0.936906 epochs 1000 learning rate 5e-05
AUC 0.777607352272

Train3
Accuracy: 0.938049 epochs 1000 learning rate 5e-05
AUC 0.806321381623

@author: botpi
'''

import tensorflow as tf
import numpy as np
import scipy.io
from apiepi import *

print "begin"
group = "train_1"
#group = "pp1"
images, labels, names = read_images(group)

learning_rate = 0.00005
training_epochs = 1000
display_step = 100

x = tf.placeholder(tf.float32, [None, 256])
y = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([256,2]))
b = tf.Variable(tf.zeros([2]))

pred = tf.nn.softmax(tf.matmul(x, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred + 1e-20)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        _, c = sess.run([optimizer, cost], feed_dict={x: images, y: labels})
        if (epoch+1) % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "Accuracy:", accuracy.eval({x:images, y: labels}), "epochs", training_epochs, "learning rate", learning_rate

    prob = pred.eval({x:images, y: labels})
    print "AUC", auc(labels, prob)
    
    resp = {}
    resp["W"] = W.eval()
    resp["b"] = b.eval()
    #scipy.io.savemat("resp_3", resp, do_compression=True)

#     s = tf.argmax(pred, 1)
#     print s.eval(feed_dict={x:images, y: labels})

print "end"