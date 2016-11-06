'''
Created on Nov 4, 2016

@author: carlos
'''
import tensorflow as tf
import numpy as np
from apiepi import *

from matplotlib import pyplot as plt
#import matplotlib.image as mpimg
import Image

#im = plt.imread("taj_orig.png")
im = np.array(Image.open('taj_bw.png'))
#im.save('taj_bw.png')
print im.shape
plt.imshow(im)
plt.show()
#conv = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # Edge detect
a = np.zeros([3, 3, 3, 3])
a[1, 1, :, :] = -4
a[0, 1, :, :] = 1
a[1, 0, :, :] = 1
a[2, 1, :, :] = 1
a[1, 2, :, :] = 1


# a[0, 0, 0, 1] = 0
#a[0, 0, 1, 0] = 1 # red

#a[0, 0, 1, 1] = 1 # green
#a[0, 1, 1, 1] = 1 # green
# a[1, 0, 2, 1] = 1 # green
# a[1, 1, 2, 1] = 1 # green
# a[1, 2, 2, 1] = 1 # green
#a[2, 2, 0, 0] = 1

# a[a, b, d, c] c=0 red. c=1 green, c=2 blue; d=0 light, d=1 normal, d=2 dark; a, b doesn't matter

#a[red, green, blue]

#a[0, 0, 1, 2] = 1 # dark blue
#a[0, 0, 2, 1] = 1
# 
print a
# print a.shape


shape = im.shape
im = np.reshape(im, [1, shape[0], shape[1], 3])
x = tf.placeholder(tf.float32, shape=[1, None, None, 3])
#x = tf.placeholder(tf.float32, shape=im.shape)
c = tf.get_variable('w', initializer=tf.to_float(a))
#c = tf.placeholder(tf.float32, shape=[3, 3, 3, 3])
#x_image = tf.reshape(x, [-1,28,28,1])
pred = conv2d(x, c)


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    #p = pred.eval(feed_dict={x:im, c:conv})
    p = sess.run([pred], feed_dict={x:im})

p = np.array(p[0][0])
print p.shape
plt.imshow(p)
plt.show()