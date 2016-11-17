'''
Train Convulotional

Created on Nov 9, 2016

Train_1
Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize dropout
0.690955 10 0.01 5 5 4 4 4 16 0.5
AUC 0.696629711752 patient 1

Accuracy: epochs learning rate cv1 size cv2 size cv1 channels cv2channels hidden img resize dropout layers
1.0 10000 0.01 5 5 4 4 4 16 0.5 3
AUC 1.0 patient 1

Train_2


Train3


@author: botpi
'''
import tensorflow as tf
import numpy as np
from epinn31 import *
import scipy.io
from epinn24 import *
from params import param

print "begin"
patient = 1
group = "train %s_new" % patient
parameters = param(patient)
training_epochs = 10000

images, labels, names = read_images(group)
features, prob, acc = train_tf(images, labels, parameters, training_epochs=training_epochs)

print "Accuracy:", "epochs", "learning rate", "cv1 size", "cv2 size", "cv1 channels", "cv2channels", "hidden", "img resize", "dropout", "layers"
print acc, training_epochs, parameters["learning_rate"], parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"], parameters["cv2_channels"], parameters["hidden"], parameters["img_resize"], parameters["dropout"], 3
print "AUC", auc(labels, prob), "patient", patient

scipy.io.savemat("resp_%s_new" % patient, features, do_compression=True)    
print "end"