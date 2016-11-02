'''
convulotional

Created on Nov 2, 2016

Train_1
Accuracy: 0.905457 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.935988185123

Train_2
Accuracy: 0.936488 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.891772920461

Train3
Accuracy: 0.942774 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.948229352347

@author: botpi
'''

import tensorflow as tf
import numpy as np
from apiepi import *

print "begin"
group = "train_1 nz"
#group = "pp1"
images, labels, names = read_images(group)

learning_rate = 0.0011
training_epochs = 20000
display_step = 100
hidden = 10

x = tf.placeholder(tf.float32, [None, 256])
y = tf.placeholder(tf.float32, [None, 2])

W1 = tf.Variable(tf.zeros([256,hidden]))
b1 = tf.Variable(tf.zeros([hidden]))
W2 = tf.Variable(tf.zeros([hidden,2]))
b2 = tf.Variable(tf.zeros([2]))

x1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
pred = tf.nn.softmax(tf.matmul(x1, W2) + b2)

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
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc = accuracy.eval({x:images, y: labels})
    print "Accuracy:", acc, "epochs", training_epochs, "learning rate", learning_rate, "hidden", hidden

    prob = pred.eval({x:images, y: labels})
    print "AUC", auc(labels, prob)
    
    resp = {}
    resp["W1"] = W1.eval()
    resp["b1"] = b1.eval()
    resp["W2"] = W2.eval()
    resp["b2"] = b2.eval()
    scipy.io.savemat("resp_1", resp, do_compression=True)

print "end"
