'''
convolutional

Created on Nov 2, 2016

Train_1
Accuracy: 0.905457 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.935988185123

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.885473  10000     0.0005         5         5         4         4         4         16
AUC 0.889153966257

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.885473 10000 0.0005 5 5 4 4 4 16
AUC 0.959253821775

Train_2
Accuracy: 0.936488 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.891772920461

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.392231 10 0.0005 5 5 4 4 4 16
AUC 0.650672905526

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.936061 10000 0.0005 5 5 4 4 4 16
AUC 0.924193989071

Train3
Accuracy: 0.942774 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.948229352347

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.936061 10 0.0005 5 5 4 4 4 16
AUC 0.540727079539

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.972849 10000 0.0005 5 5 4 4 4 16
AUC 0.978215983363

@author: botpi
'''

import tensorflow as tf
import numpy as np
from apiepi import *

print "begin"
group = "train_1 nz"
#group = "pp1"
images, labels, names = read_images(group)

learning_rate = 0.0005
training_epochs = 10000
display_step = 100
cv1_size = 5
cv2_size = 5
cv1_channels = 4
cv2_channels = 4
hidden = 4
img_resize = 16

x = tf.placeholder(tf.float32, shape=[None, 256])
y = tf.placeholder(tf.float32, shape=[None, 2])  
  
# First Convolutional Layer  
W_conv1 = weight_variable([cv1_size, cv1_size, 1, cv1_channels])
b_conv1 = bias_variable([cv1_channels])

x_image = tf.reshape(x, [-1,img_resize,img_resize,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print h_conv1
print h_pool1

# Second Convolutional Layer
W_conv2 = weight_variable([cv2_size, cv2_size, cv1_channels, cv2_channels])
b_conv2 = bias_variable([cv2_channels])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print h_conv2
print h_pool2

# Densely Connected Layer
W_fc1 = weight_variable([img_resize/4 * img_resize/4 * cv2_channels, hidden])
b_fc1 = bias_variable([hidden])

h_pool2_flat = tf.reshape(h_pool2, [-1, img_resize/4 * img_resize/4  * cv2_channels])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([hidden, 2])
b_fc2 = bias_variable([2])

pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-20), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
#         print epoch
        _, c = sess.run([optimizer, cost], feed_dict={x: images, y: labels, keep_prob: 0.5})
        #if (epoch+1) % display_step == 0:
        print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
    acc = accuracy.eval({x:images, y: labels, keep_prob: 1})
    print "Accuracy:", "epochs", "learning rate", "cv1 size", "cv2 size", "cv1 channels", "cv2channels", "hidden", "img resize"
    print acc, training_epochs, learning_rate, cv1_size, cv2_size, cv1_channels, cv2_channels, hidden, img_resize

    prob = pred.eval({x:images, y: labels, keep_prob: 1})
    print "AUC", auc(labels, prob)
    
    resp = {}
    resp["W_conv1"] = W_conv1.eval()
    resp["b_conv1"] = b_conv1.eval()
    resp["W_conv2"] = W_conv2.eval()
    resp["b_conv2"] = b_conv2.eval()
    resp["W_fc1"] = W_fc1.eval()
    resp["b_fc1"] = b_fc1.eval()
    resp["W_fc2"] = W_fc2.eval()
    resp["b_fc2"] = b_fc2.eval()
    scipy.io.savemat("resp_1", resp, do_compression=True)

print "end"
