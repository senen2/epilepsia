'''
Envia y recibe una imagen a tensorflow
Created on Nov 4, 2016

@author: carlos
'''
import tensorflow as tf
import numpy as np
from apiepi import *

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

im = mpimg.imread("cb.png")
print im.shape
# plt.imshow(im)
# plt.show()

x = tf.placeholder(tf.float32, shape=im.shape)
pred = x * 2
rk = tf.rank(pred)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    p = pred.eval(feed_dict={x:im})
    r = rk.eval(feed_dict={x:im})

print r
plt.imshow(p / 2)
plt.show()