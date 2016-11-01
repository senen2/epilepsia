'''
Manually Selecting threshold for correlation coefficients

Created on 14/10/2016

train1
F1 epsilon 0.284745762712 0.185 tp fp fn 84 357 65

train2


train3


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
   
epsilon = .185
channels = 2
while True:
    ep = raw_input("epsilon %s  " % epsilon)
    if ep!="":
        epsilon = eval(ep)

    cha = raw_input("channels %s  " % channels)
    if cha!="":
        channels = eval(cha)

    bestF1 = 0
    bestEpsilon = 0
    tp = 0
    fp = 0
    fn = 0
    for d in t:
        pval = np.abs(t[d]["corr"][0][0]) < epsilon
        yval = d[-1]
        
        if np.sum(pval) > channels**2:
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
