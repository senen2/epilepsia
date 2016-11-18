'''
Train Convulotional

Created on Nov 9, 2016

Train_1


Train_2
epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_resize dropout
    1000      0.01        5       5          4            4        4        16      0.5
Accuracy: 0.893004 Accuracy test: 0.891626
AUC 0.98064516129 AUC test 0.901635484681 patient 2

epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_resize dropout
    1000      0.01        5       5          4            4        4        16      0.5
Accuracy: 0.893004 Accuracy test: 0.891626
AUC 0.954810350939 AUC test 0.88105851331 patient 2

epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_resize dropout
    1000      0.01        5       5          4            8        4        16      0.5
Accuracy: 0.893004 Accuracy test: 0.891626
AUC 0.96841191067 AUC test 0.890821195379 patient 2

epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_resize dropout
    1000      0.01        5       5          16            8        4        16      0.5
Accuracy: 0.893004 Accuracy test: 0.891626
AUC 0.976809641971 AUC test 0.912583186841 patient 2

epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_resize dropout
    1000      0.01        5       5          8            16        4        16      0.5
Accuracy: 0.893004 Accuracy test: 0.891626
AUC 0.965171924849 AUC test 0.899563661477 patient 2

epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_resize dropout
    1000      0.01        5       5          4            4        8        16      0.5
Accuracy: 0.893004 Accuracy test: 0.891626
AUC 0.958156682028 AUC test 0.901203854847 patient 2

epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_resize dropout
    1000      0.01        5       5          4            4        5        16      0.5
Accuracy: 0.893004 Accuracy test: 0.891626
AUC 0.942141084722 AUC test 0.875761238071 patient 2

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
patient = 3
group = "train %s_new" % patient
parameters = param(patient)
training_epochs = 10000

images, labels, names = read_images(group)
train_images, train_labels, test_images, test_labels = read_train_test(group, .6)

features, train_prob, train_acc, test_prob, test_acc = train_tf(train_images, train_labels, test_images, test_labels, parameters, training_epochs=training_epochs)

print "epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_resize dropout"
print ("    %s      %s        %s       %s          %s            %s        %s        %s      %s" 
    % (training_epochs, parameters["learning_rate"], parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"]
       , parameters["cv2_channels"], parameters["hidden"], parameters["img_resize"], parameters["dropout"]))
print "Accuracy:", train_acc, "Accuracy test:", test_acc
print "AUC", auc(train_labels, train_prob), "AUC test", auc(test_labels, test_prob), "patient", patient

scipy.io.savemat("resp_%s_new" % patient, features, do_compression=True)    
print "end"