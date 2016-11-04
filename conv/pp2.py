'''
Created on Nov 4, 2016

@author: carlos
'''
import tensorflow as tf
import numpy as np
from apiepi import *

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

im = mpimg.imread("cb.png")
conv = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # Edge detect

im = tf.reshape(im, [-1, 323, 272, 3])
x = tf.placeholder(tf.float32, shape=[1, 323, 272, 3])
#x = tf.placeholder(tf.float32, shape=im.shape)
c = tf.placeholder(tf.float32, shape=conv.shape)
pred = x * 2
#x_image = tf.reshape(x, [-1,28,28,1])
y = conv2d(x, c)


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    p = pred.eval(feed_dict={x:im, c:conv})

plt.imshow(p / 2)
plt.show()