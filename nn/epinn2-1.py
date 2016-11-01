'''
Calculate W1, b1, W2, b2 and save

Created on Oct 21, 2016

Train_1
Accuracy: 0.890292 epochs 1000 learning rate 5e-05
AUC 0.776613005451

Accuracy: 0.895028 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.935543161364

Accuracy: 0.897395 epochs 20000 learning rate 0.0011 hidden 20
AUC 0.895816960325

Accuracy: 0.897395 epochs 20000 learning rate 0.0011 hidden 30
AUC 0.938071998986

Train_2
Accuracy: 0.936906 epochs 1000 learning rate 5e-05
AUC 0.777607352272

Accuracy: 0.937338 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.893980943234

Accuracy: 0.936041 epochs 20000 learning rate 0.0011 hidden 20
AUC 0.893093186069

Accuracy: 0.937338 epochs 20000 learning rate 0.0011 hidden 30
AUC 0.892575064449

Train3
Accuracy: 0.938049 epochs 1000 learning rate 0.0011 hidden 10 
AUC 0.858617653798

Accuracy: 0.942654 epochs 10000 learning rate 0.0011 hidden 10
AUC 0.917758119565

Accuracy: 0.944747 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.960810207798

Accuracy: 0.939724 epochs 1000 learning rate 0.0011 hidden 20
AUC 0.857701074569

Accuracy: 0.939724 epochs 2000 learning rate 0.0011 hidden 20
AUC 0.879593448871

Accuracy: 0.942235 epochs 3000 learning rate 0.0011 hidden 20
AUC 0.890595414692

Accuracy: 0.942235 epochs 4000 learning rate 0.0011 hidden 20
AUC 0.897915988277

Accuracy: 0.943491 epochs 8000 learning rate 0.0011 hidden 20
AUC 0.912952108735

Accuracy: 0.944328 epochs 10000 learning rate 0.0011 hidden 20
AUC 0.917061639953

Accuracy: 0.943072 epochs 20000 learning rate 0.0011 hidden 20
AUC 0.929103802598

Accuracy: 0.939305 epochs 1000 learning rate 0.0011 hidden 30
AUC 0.85620258813

Accuracy: 0.939724 epochs 2000 learning rate 0.0011 hidden 30
AUC 0.878408529011

Accuracy: 0.94684 epochs 20000 learning rate 0.0011 hidden 30
AUC 0.945276601903

@author: botpi
'''

import tensorflow as tf
import numpy as np
import scipy.io
from apiepi import *

print "begin"
group = "train_1 nz"
#group = "pp1"
images, labels, names = read_images(group)

learning_rate = 0.0011
training_epochs = 200
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
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "Accuracy:", accuracy.eval({x:images, y: labels}), "epochs", training_epochs, "learning rate", learning_rate, "hidden", hidden

    prob = pred.eval({x:images, y: labels})
    print "AUC", auc(labels, prob)
    
    resp = {}
    resp["W1"] = W1.eval()
    resp["b1"] = b1.eval()
    resp["W2"] = W2.eval()
    resp["b2"] = b2.eval()
    print resp["W1"].shape, resp["b1"].shape, resp["W2"].shape, resp["b2"].shape, images.shape
    print resp["b1"]
    #scipy.io.savemat("resp_1", resp, do_compression=True)

#     s = tf.argmax(pred, 1)
#     print s.eval(feed_dict={x:images, y: labels})

print "end"