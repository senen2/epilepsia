'''
Calculate W1, b1, W2, b2 with train and test and save

Created on Oct 21, 2016

Train_1
Accuracy: 0.890292 epochs 1000 learning rate 5e-05
AUC 0.776613005451

Train_2
Accuracy: 0.936906 epochs 1000 learning rate 5e-05
AUC 0.777607352272

Train3
Accuracy: 0.934449 epochs 1000 learning rate 0.0011 hidden 10
AUC 0.856652905684

Accuracy: 0.934449 epochs 1000 learning rate 0.0011 hidden 20
AUC 0.855350905049

Accuracy: 0.934449 epochs 1000 learning rate 0.002 hidden 20 *
AUC 0.857732613528

Accuracy: 0.934449 epochs 20000 learning rate 0.0011 hidden 20
AUC 0.877770720864

Accuracy: 0.934449 epochs 20000 learning rate 0.002 hidden 20
AUC 0.881867259447

Accuracy: 0.934449 epochs 1000 learning rate 0.0011 hidden 30
AUC 0.855795490632

Accuracy: 0.924686 epochs 20000 learning rate 0.0011 hidden 30
AUC 0.88532867577

Accuracy: 0.920502 epochs 20000 learning rate 0.0011 hidden 40
AUC 0.87554779295

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
n = np.int(images.shape[0]*.7)
train_images = images[:n]
train_labels = labels[:n]
test_images = images[n:]
test_labels = labels[n:]
print train_images.shape, train_labels.shape, test_images.shape, test_labels.shape
learning_rate = 0.002
training_epochs = 1000
display_step = 100
hidden = 20

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
    scipy.io.savemat("resp_4", resp, do_compression=True)

#     s = tf.argmax(pred, 1)
#     print s.eval(feed_dict={x:images, y: labels})

print "end"