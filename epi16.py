'''
Plot ROC curve

Created on 14/09/2016

all best F1 epsilon anomalies 0.268138801262 1.5e-34 24600         pval > epsilon
all best F1 epsilon anomalies 0.269158878505 1.5e-34 101500, tp fp fn 72 314 77        pval > epsilon

all best F1 epsilon anomalies 0.221445221445 1.5e-34 6500 tp fp fn 95 614 54 pval < epsilon

@author: botpi
'''
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

print "reading..."

group = "train_1"
#group = "pp1"
t = scipy.io.loadmat(group)
t.pop("__globals__")
t.pop("__version__")
t.pop("__header__")

epsilon = 1.5e-34           # epsilon=1.2e-34, anom=28000 F1=0.266
anom = 101500

y = []  # true-positive rate is also known as sensitivity, recall or probability of detection
x = []  # false-positive rate is also known as the fall-out or probability of false alarm

tp = 0
fp = 0
fn = 0

for d in t:
    pval = t[d][0]
    yval = d[-1]
    an = np.sum(pval > epsilon)
    #print "anomalies", d, an
    tp1 = 0
    fp1 = 0
    if an > anom:
        if yval == "1":
            tp += 1
            tp1 = 1
        else:
            fp += 1
            fp1 = 1
    elif yval == "1":
        fn += 1

    y.append(tp1)
    x.append(fp1)

auc = np.trapz(y,x)
print "Area Under Curve", auc

plt.plot(x,y)
plt.show()
