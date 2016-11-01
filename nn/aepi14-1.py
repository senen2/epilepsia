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

epsilon 0.23 , F1 tp fp fn (0.2771929824561403, 79, 342, 70)
AUC 0.616663171141

epsilon 0.24 , F1 tp fp fn (0.2764505119453925, 81, 356, 68)
AUC 0.617298191648

train2
F1 epsilon 0.138486312399 0.1 tp fp fn 129 1584 21

epsilon 0.1 , F1 tp fp fn (0.13811563169164884, 129, 1589, 21)
AUC 0.56820582878

train3
F1 epsilon 0.172437202987 0.45 tp fp fn 127 1196 23

epsilon 0.45 , F1 tp fp fn (0.17243720298710116, 127, 1196, 23)
AUC 0.656844919786

@author: botpi
'''
import numpy as np
import scipy.io
from apiepi import *

print "reading..."

group = "train_3 nz"
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
f1 = F1()
   
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
    f1.init()
    prob = []
    labels = []
    for d in t:
        pval = t[d]["corr"][0][0][channelx][channel]
        yval = d[-1]
        
        f1.take(pval < epsilon, yval)
        
        if pval < epsilon:
            prob.append([0, 1])
        else:
            prob.append([1, 0])

        if yval == "1":
            labels.append([0, 1])
        else:
            labels.append([1, 0])

    if f1.calc() > bestF1:
        bestF1 = f1.calc()
        bestEpsilon = epsilon
        besttp = f1.tp
        bestfp = f1.fp
        bestfn = f1.fn
        
    labels = np.array(labels)
    prob = np.array(prob)
    print "F1 epsilon", bestF1, epsilon, "tp fp fn", besttp, bestfp, bestfn
    print "epsilon", epsilon, ", F1 tp fp fn", F1c(labels, prob)
    print "AUC", auc(labels, prob)
                
#     print "Best F1 epsilon", bestF1,  bestEpsilon, "tp fp fn",bestF1, besttp, bestfp, bestfn
#     
#     if bestF1 > allbestF1:
#         allbestF1 = bestF1
#         allbestEpsilon = bestEpsilon
#     
#     print "all best F1 epsilon", allbestF1, allbestEpsilon
    
    print

print "end"
