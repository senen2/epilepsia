'''
Plot probabilites

Created on 14/09/2016

@author: papi
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.io, scipy.stats
from matplotlib import cm

def visualizeFit(X, mu, sigma2):
    X1, X2 = np.meshgrid(np.arange(-400, 600, 10), np.arange(-400, 500, 10))
    a = np.array([[x, y] for x,y in zip(np.ravel(X1), np.ravel(X2))])
    
    Z = scipy.stats.multivariate_normal.pdf(a, mu, sigma2)
    Z = np.reshape(Z, X1.shape)     

    # Do not plot if there are infinities
    if np.sum(np.isinf(Z)) == 0:
        #cs = plt.contourf(X1, X2, Z, zdir='y', cmap=cm.prism)
        cs = plt.contourf(X1, X2, Z, zdir='y', cmap=cm.prism, levels=[1.0000e-020,  1.0000e-017,  1.0000e-014,  1.0000e-011,  1.0000e-008,  1.0000e-005, 0.0001])
        proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
                 for pc in cs.collections]
 
        plt.legend(proxy, ["1e-20", "1e-17", "1e-14", "1e-11", "1e-08", "1e-5", "0.0001"])


group = "train_1"
file = "1_66_1"
directory = "c:/concursos/epilepsia/%s/" % group
mat = scipy.io.loadmat(directory + file + ".mat")
t = mat['dataStruct'][0][0][0]
X = t[:, 1:3]

mu = np.mean(X, axis=0)
#sigma2 = np.std(X, axis=0)**2
cov = np.cov(X, rowvar=0)
corr= np.corrcoef(X, rowvar=0)
print "mu", mu
print "cov", cov
print "corrcoef", corr
#print scipy.stats.multivariate_normal.pdf(mu, mu, cov)

plt.plot(X[:,0], X[:,1], linestyle="", marker='o')
visualizeFit(X, mu, cov)

# plt.hist(X[:,0], bins="auto", color="blue")
# plt.hist(X[:,1], bins="auto", color="red")
#plt.hist(t, bins="auto")

# plt.plot(X[:,0], color="blue")
# plt.plot(X[:,1], color="red")

#pval = scipy.stats.multivariate_normal.pdf(X, mu, cov)
#plt.hist(pval, bins="auto")

fig = plt.gcf()
fig.canvas.set_window_title(file)
plt.show()
