'''
Created on Oct 23, 2016

@author: carlos
'''
import tensorflow as tf
import numpy as np

a = np.array([[1,2,3,4,5]])
b = np.transpose(a)

print a.dot(b)

x = tf.placeholder(tf.float32, [None, 5])
y = tf.placeholder(tf.float32, [None, 5])

c = tf.matmul(x, y, transpose_b=True)

output = tf.Print(c, [c], "output")

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
d = sess.run(output, feed_dict={x: a, y: a})
print "final", d.shape, d
