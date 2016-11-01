'''
Save results

Created on 14/09/2016

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

print "begin..."

epsilon = [.3, .04, .3]

x = []
x.append(["File", "Class"])

for i in range(3):
    group = "test_%s" % (i+1) 
    print i, epsilon[i], group
    t = scipy.io.loadmat(group)
    t.pop("__globals__")
    t.pop("__version__")
    t.pop("__header__")

    p = 0
    for d in t:
        pval = t[d]["corr"][0][0][0][1]
        v = 0
        if np.abs(pval) < epsilon[i]:
            v = 1
            p += 1
    
        x.append([d + ".mat", v])
    print "positivos", p

#np.savetxt("submission_12.csv", x, delimiter=',', fmt="%s,%s")
#x = np.array(x)
#x.tofile(group + ".csv", sep=',', format="%s,%s")
print "end"