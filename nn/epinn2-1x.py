'''
Created on Oct 21, 2016

Train_1
Accuracy: 0.893449 epochs 255 learning rate 0.0005
Accuracy: 0.890292 epochs 1000 learning rate 5e-05

@author: botpi
'''

import tensorflow as tf
import numpy as np
import scipy.io
from apiepi import *

print "begin"
group = "train_1"
#group = "pp1"
images, labels = read_images(group)

learning_rate = 0.00005
training_epochs = 1000
display_step = 100

x = tf.placeholder(tf.float32, [None, 256])
y = tf.placeholder(tf.float32, [None, 2])

W1 = tf.Variable(tf.zeros([256,20]))
b1 = tf.Variable(tf.zeros([20]))
W2 = tf.Variable(tf.zeros([20,2]))
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
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "Accuracy:", accuracy.eval({x:images, y: labels}), "epochs", training_epochs, "learning rate", learning_rate
    
    resp = {}
    resp["W1"] = W1.eval()
    resp["b1"] = b1.eval()
    resp["W2"] = W2.eval()
    resp["b2"] = b2.eval()
    scipy.io.savemat("resp_1-1", resp, do_compression=True)

print "end"