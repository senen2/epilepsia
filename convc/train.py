'''
Train Convulotional

Created on Nov 9, 2016

Train_1
Accuracy: 0.905457 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.935988185123

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.885473   10000     0.0005         5         5         4         4         4         16
AUC 0.959253821775

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.998744 10000 0.0005 5 5 4 4 4 16
AUC 0.999992609017 patient 1

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.835427 1000 0.0005 5 5 4 4 4 16 .5
AUC 0.945373244642 patient 1

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize dropout
0.913317 1000 0.0005 5 5 4 4 4 16 .3
AUC 0.959563932003 patient 1

Train_2
Accuracy: 0.936488 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.891772920461

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.936061  10000     0.0005         5         5         4         4         4         16
AUC 0.924193989071

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.892452 1000 0.0005 5 5 4 4 4 16
AUC 0.748961613949

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.892452 10000 0.0005 5 5 4 4 4 16
AUC 0.5 patient 2

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.892452 1000 0.005 5 5 4 4 4 16
AUC 0.944406408325 patient 2

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize *
0.981746     1000     0.01         5         5         4         4         4         16
AUC 0.994771301495 patient 2

Train3
Accuracy: 0.942774 epochs 20000 learning rate 0.0011 hidden 10 *
AUC 0.948229352347

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.972849  10000     0.0005         5         5         4         4         4         16
AUC 0.978215983363

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize *
0.882762 10000 0.0005 5 5 4 4 4 16
AUC 0.990899754132 patient 3

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize
0.991196 10000 0.005 5 5 4 4 4 16
AUC 0.99501312336 patient 3

@author: botpi
'''
import tensorflow as tf
import numpy as np
from epinn31 import *
import scipy.io
from epinn24 import *

print "begin"
patient = 3
group = "train %s_new" % patient
parameters = {  "cv1_size": 5
              , "cv2_size": 5
              , "cv1_channels": 4
              , "cv2_channels": 4
              , "hidden": 4
              , "img_resize": 16}
learning_rate = 0.0005
dropout = 0.3
training_epochs = 10000

images, labels, names = read_images(group)
features, prob, acc = train_tf(images, labels, parameters, learning_rate=learning_rate, training_epochs=training_epochs, dropout=dropout)

print "Accuracy:", "epochs", "learning rate", "cv1 size", "cv2 size", "cv1 channels", "cv2channels", "hidden", "img resize", "dropout"
print acc, training_epochs, learning_rate, parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"], parameters["cv2_channels"], parameters["hidden"], parameters["img_resize"], dropout
print "AUC", auc(labels, prob), "patient", patient

scipy.io.savemat("resp_%s_new" % patient, features, do_compression=True)    
print "end"