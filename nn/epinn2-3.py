'''
Calculate W1, b1, W2, b2 with train and test splitting 1 and 0 separately, and save

Created on Oct 21, 2016

Train_1
Accuracy: 0.890292 epochs 1000 learning rate 5e-05
AUC 0.776613005451

Accuracy: 0.88491 epochs 1000 learning rate 5e-05 hidden 10
AUC 0.658831085421

Accuracy: 0.877238 epochs 20000 learning rate 5e-05 hidden 10
AUC 0.645022479127

Accuracy: 0.88491 epochs 1000 learning rate 5e-05 hidden 15
AUC 0.65838150289

Accuracy: 0.872123 epochs 10000 learning rate 0.0005 hidden 15 *
AUC 0.660179833012

Train_2
Accuracy: 0.93608 epochs 10000 learning rate 0.0005 hidden 15 *
AUC 0.84892935424

Train3
Accuracy: 0.93185 epochs 10000 learning rate 0.0005 hidden 15 *
AUC 0.903000329707

@author: botpi
'''

import tensorflow as tf
import numpy as np
import scipy.io
from apiepi import *

print "begin"
group = "train_3 nz"
#group = "pp1"
images, labels, _ = read_images(group)
train_images, train_labels, test_images, test_labels = read_train_test(group, .7)

learning_rate = 5e-04
training_epochs = 10000
display_step = 100
hidden = 15

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
        _, c = sess.run([optimizer, cost], feed_dict={x: train_images, y: train_labels})
        if (epoch+1) % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    acc = accuracy.eval({x:test_images, y: test_labels})
    print "Accuracy:", acc, "epochs", training_epochs, "learning rate", learning_rate, "hidden", hidden

    prob = pred.eval({x:test_images, y: test_labels})
    print "AUC", auc(test_labels, prob)
    
    resp = {}
    resp["W1"] = W1.eval()
    resp["b1"] = b1.eval()
    resp["W2"] = W2.eval()
    resp["b2"] = b2.eval()
    scipy.io.savemat("resp_3", resp, do_compression=True)

print "end"
