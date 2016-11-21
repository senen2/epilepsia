'''
Create submission

Created on Nov 9, 2016

@author: botpi
'''
import tensorflow as tf
import numpy as np
from epinn31 import *
import scipy.io
from apiepi import *
from params import param

print "begin"
group = "../data/train"
#group = "test"
sub_file = "submission_conv2_6.csv"
    
r = []
r.append(["File", "Class"])

for i in range(3):
    patient = i+1
    features = scipy.io.loadmat("resp_%s_new" % patient)
    #images, labels, names = read_images("%s %s_new" % (group, ii))
    images, labels, names = read_images("%s_%s_new" % (group, patient))
    parameters = param(patient)
    
    prob = eval_conv(images, parameters, features)
    
    p = 0
    for i in xrange(len(names)):
        r.append([names[i] + ".mat", prob[i][1]])
        if prob[i][0] < prob[i][1]:
            p += 1

    print "patient", patient, "positives", p, "totals", len(names)
    if group == "../data/train":
        print "AUC", auc(labels, prob)
    else:
        np.savetxt(sub_file, r, delimiter=',', fmt="%s,%s")

print "gran total", len(r)
print "end"