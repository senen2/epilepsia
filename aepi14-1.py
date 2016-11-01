'''
Manually Selecting threshold and channel for correlation coefficients

Created on 14/10/2016

train1
F1 epsilon 0.273789649416 0.25 tp fp fn 82 368 67

train2


train3


without zeros

train1
F1 epsilon 0.273789649416 0.25 tp fp fn 82 368 67

train2
F1 epsilon 0.138486312399 0.1 tp fp fn 129 1584 21

train3
F1 epsilon 0.172437202987 0.45 tp fp fn 127 1196 23

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

group = "train_1"
#group = "pp1"
t = scipy.io.loadmat(group)
t.pop("__globals__")
t.pop("__version__")
t.pop("__header__")

allbestEpsilon = 0
allbestF1 = 0
allbestanom = 0
besttp = 0 
bestfp = 0
bestfn = 0
   
epsilon = .25
channel = 1
channelx = 0
while True:
    ep = raw_input("epsilon %s  " % epsilon)
    if ep!="":
        epsilon = eval(ep)

    ch = raw_input("channel %s  " % channel)
    if ch!="":
        channel = eval(ch)

    chx = raw_input("channelx %s  " % channelx)
    if chx!="":
        channelx = eval(chx)

    bestF1 = 0
    bestEpsilon = 0
    tp = 0
    fp = 0
    fn = 0
    for d in t:
        pval = t[d]["corr"][0][0][channelx][channel]
        yval = d[-1]

        if np.abs(pval) < epsilon:
            if yval == "1":
                tp += 1
            else:
                fp += 1
        elif yval == "1":
            #print d
            fn += 1

    F1 = calcF1(tp, fp, fn)
    if F1 > bestF1:
        bestF1 = F1
        bestEpsilon = epsilon
        besttp = tp
        bestfp = fp
        bestfn = fn
        
    print "F1 epsilon", bestF1, epsilon, "tp fp fn", besttp, bestfp, bestfn
                
#     print "Best F1 epsilon", bestF1,  bestEpsilon, "tp fp fn",bestF1, besttp, bestfp, bestfn
#     
#     if bestF1 > allbestF1:
#         allbestF1 = bestF1
#         allbestEpsilon = bestEpsilon
#     
#     print "all best F1 epsilon", allbestF1, allbestEpsilon
    
    print

print "end"
