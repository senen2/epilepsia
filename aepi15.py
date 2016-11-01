'''
Calculus of F1 for all negatives

Created on 14/09/2016

@author: botpi
'''
import numpy as np
import scipy.io

def calcF1(tp, fp, fn):
    if tp+fp > 0:
        prec = tp / float(tp + fp)
    else:
        prec = 0
    
    if tp+fn > 0:
        rec = tp / float(tp + fn)
    else:
        rec = 0
    
    if rec+prec > 0:
        F1 = 2 * prec * rec / (prec + rec)
    else:
        F1 = 0
    
    return F1

print "reading..."

group = "train_3"
#group = "pp1"
t = scipy.io.loadmat(group)
t.pop("__globals__")
t.pop("__version__")
t.pop("__header__")

n = 0
pos = 0
neg = 0
tp = 0
fp = 0
fn = 0
for d in t:
    yval = d[-1]
    if yval == "0":
        neg += 1
    else:
        pos += 1 

    tn = neg
    fn = pos

F1 = calcF1(tp, fp, fn)
    
print "F1 neg pos total ", F1, neg, pos, neg+pos
print "end"
