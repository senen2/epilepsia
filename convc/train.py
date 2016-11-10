'''
Train Convulotional

Created on Nov 9, 2016

Train_1
Accuracy: 0.905457 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.935988185123

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.885473   10000     0.0005         5         5         4         4         4         16
AUC 0.959253821775

Train_2
Accuracy: 0.936488 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.891772920461

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.936061  10000     0.0005         5         5         4         4         4         16
AUC 0.924193989071

Train3
Accuracy: 0.942774 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.948229352347

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.972849  10000     0.0005         5         5         4         4         4         16
AUC 0.978215983363

@author: botpi
'''
import tensorflow as tf
import numpy as np
from epinn31 import *
import scipy.io
from epinn24 import *

print "begin"
iid = 1
group = "train_%s nz" % iid
parameters = {  "cv1_size": 5
              , "cv2_size": 5
              , "cv1_channels": 4
              , "cv2_channels": 4
              , "hidden": 4
              , "img_resize": 16}
learning_rate = 0.0005
training_epochs = 10

images, labels, names = read_images(group)
features, prob, acc = train_tf(images, labels, parameters, learning_rate = learning_rate, training_epochs = training_epochs)

print "Accuracy:", "epochs", "learning rate", "cv1 size", "cv2 size", "cv1 channels", "cv2channels", "hidden", "img resize"
print acc, training_epochs, learning_rate, parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"], parameters["cv2_channels"], parameters["hidden"], parameters["img_resize"]
print "AUC", auc(labels, prob)

scipy.io.savemat("resp_%s" % iid, features, do_compression=True)    
print "end"