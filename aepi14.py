'''
Manually Selecting threshold Epsilon and anomalies

Created on 14/09/2016

train1
all best F1 epsilon anomalies 0.268138801262 1.5e-34 24600         pval > epsilon
all best F1 epsilon anomalies 0.269158878505 1.5e-34 101500, tp fp fn 72 314 77        pval > epsilon

all best F1 epsilon anomalies 0.221445221445 1.5e-34 6500 tp fp fn 95 614 54 pval < epsilon

train2
F1 epsilon 0.127557160048 6e-36 anomalies 103000 tp fp fn 53 628 97

train3
F1 epsilon 0.1592039801 1.5e-34 anomalies 15000 tp fp fn 48 405 102

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

group = "train_1c"
group = "pp1c"
t = scipy.io.loadmat(group)
t.pop("__globals__")
t.pop("__version__")
t.pop("__header__")

allbestEpsilon = 0
allbestF1 = 0
allbestanom = 0
   
epsilon = 1.5e-34           # epsilon=1.2e-34, anom=28000 F1=0.266
anomf = 101400
anomt = 101600
stepanom = 100
while True:
    ep = raw_input("epsilon ")
    if ep!="":
        epsilon = eval(ep)
    anf = raw_input("anom from ")
    if anf!="":
        anomf = eval(anf)
    ant = raw_input("anom to ")
    if ant!="":
        anomt = eval(ant)
    st = raw_input("anom step ")
    if st!="":
        stepanom = eval(st)

    print epsilon, anomf, anomt, stepanom
    bestF1 = 0
    bestEpsilon = 0
        
    for anom in xrange(anomf, anomt, stepanom):
        tp = 0
        fp = 0
        fn = 0
        for d in t:
            pval = t[d][0]
            yval = d[-1]
            an = np.sum(pval > epsilon)
            #print "anomalies", d, an
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
            bestanom = anom
        
        print "F1 epsilon", bestF1, epsilon, "anomalies", anom, "tp fp fn", besttp, bestfp, bestfn
                
    print "Best F1 epsilon", bestF1,  bestEpsilon, "anomalies", bestanom, "tp fp fn",bestF1, besttp, bestfp, bestfn
    
    if bestF1 > allbestF1:
        allbestF1 = bestF1
        allbestEpsilon = bestEpsilon
        allbestanom = bestanom 
    
    print "all best F1 epsilon anomalies", allbestF1, allbestEpsilon, allbestanom
    
    print

print "end"
