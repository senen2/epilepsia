'''
Create submission

Created on Nov 9, 2016

@author: botpi
'''
import tensorflow as tf
import numpy as np
from epinn31 import *
import scipy.io

print "begin"
group = "test"
sub_file = "submission_conv_5.csv"
parameters = {  "cv1_size": 5
              , "cv2_size": 5
              , "cv1_channels": 4
              , "cv2_channels": 4
              , "hidden": 4
              , "img_resize": 16}
    
r = []
r.append(["File", "Class"])

for i in range(3):
    ii = i+1
    features = scipy.io.loadmat("resp_%s_new" % ii)
    #images, labels, names = read_images("%s %s_new" % (group, ii))
    images, labels, names = read_images("%s_%s_new nz" % (group, ii))
    
    prob = eval_conv(images, parameters, features)
    
    p = 0
    for i in xrange(len(names)):
        r.append([names[i] + ".mat", prob[i][1]])
        if prob[i][0] < prob[i][1]:
            p += 1

    print "positives", p, "totals", len(names)
    if group == "train":
        print "AUC", auc(labels, prob)

print "gran total", len(r)
np.savetxt(sub_file, r, delimiter=',', fmt="%s,%s")

print "end"