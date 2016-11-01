'''
Selecting threshold Epsilon and anomalies

Created on 14/09/2016

@author: papi
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


group = "train_1"
group = "pp1"
t = scipy.io.loadmat(group)
t.pop("__globals__")
t.pop("__version__")
t.pop("__header__")

bestEpsilon = 0
bestF1 = 0

mx = 0
mi = 1e99
for d in t:
    #print d, np.min(t[d]), np.max(t[d])
    m = np.max(t[d][0])
    if m > mx:
        mx = m

    m = np.min(t[d][0])
    if m < mi and m>0:
        mi = m
    
steps = 1000
#mi = mi + (mx - mi) / steps
stepsize = (mx - mi) / steps
print mi, mx, stepsize
brk = 1
for epsilon in np.arange(mi, mx, stepsize):    
    if brk == 10:
        break
    print "#########################"
    print "epsilon #", brk, " of ", steps
    for anom in xrange(1000, 21000, 1000):
        tp = 0
        fp = 0
        fn = 0
        #print "epsilon", epsilon, anom
        for d in t:
            #print t[d].shape, t[d][0].shape
            pval = t[d][0]
            yval = d[-1]
            an = np.sum(pval > epsilon)
            print "anomalies", d, an
            if an > anom:
                if yval == "1":
                    tp += 1
                else:
                    fp += 1
            elif yval == "1":
                fn += 1

        F1 = calcF1(tp, fp, fn)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
            besttp = tp
            bestfp = fp
            bestfn = fn
        
        print "F1 epsilon", bestF1, epsilon, "anomalies", anom, "tp fp fn", besttp, bestfp, bestfn
    brk += 1
            

print "Best F1 epsilon", bestF1,  bestEpsilon, "anomalies", anom, "tp fp fn",bestF1, besttp, bestfp, bestfn
print "end"
