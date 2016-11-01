'''
Created on Oct 21, 2016

Train_1
Accuracy: 0.891081 epochs 255 learning rate 0.001
Accuracy: 0.89266 epochs 255 learning rate 0.001
Accuracy: 0.891871 epochs 255 learning rate 0.001

@author: botpi
'''

import tensorflow as tf
import numpy as np
import scipy.io
import os

class Data(object):
    def __init__(self, group):
        #directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/epilepsia/%s/" % group
        t = scipy.io.loadmat(group)
        t.pop("__globals__")
        t.pop("__version__")
        t.pop("__header__")
        
        _images = []
        _labels = []
        for d in t:
            if not np.isnan(t[d]["corr"][0][0]).any():
                _images.append(t[d]["corr"][0][0].ravel())
                if d[-1]=="1":
                    _labels.append([0, 1])
                else:
                    _labels.append([1, 0])            
        
        self._images = np.array(_images)
        self._labels = np.array(_labels)
        self._num_examples = self.images.shape[0]
        self._index_in_epoch = 0
        self._epochs_completed = 0

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed        

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.images[start:end], self.labels[start:end]

print "begin"
learning_rate = 0.001
training_epochs = 255
batch_size = 100
display_step = 100

group = "train_1"
#group = "pp1"
data = Data(group)
print data.images.shape, data.labels.shape
#z()

x = tf.placeholder(tf.float32, [None, 256])
y = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([256,2]))
b = tf.Variable(tf.zeros([2]))

pred = tf.nn.softmax(tf.matmul(x, W) + b)

# x = tf.Print(x, [x], "x: ")
# b = tf.Print(b, [b], "Bias: ")
# W = tf.Print(W, [W], "Weight: ")
# matmul_result = tf.matmul(x, W)
# matmul_result = tf.Print(matmul_result, [matmul_result], "Matmul: ")
# tf.is_nan(x, name=None)
# pred = tf.nn.softmax(matmul_result + b)



cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred + 1e-10)))
#cost = -tf.reduce_sum(y*tf.log(tf.clip_by_value(pred,1e-10,1.0)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(data.num_examples/batch_size)
        #print data.num_examples, batch_size, total_batch
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = data.next_batch(batch_size)
            #print batch_xs.shape, batch_ys.shape, batch_ys
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "Accuracy:", accuracy.eval({x:data.images, y: data.labels}), "epochs", training_epochs, "learning rate", learning_rate