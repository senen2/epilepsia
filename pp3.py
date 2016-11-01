import numpy as np
a = [1, 1, 2, 2, 2, 2, 3]
hist, bin_edges = np.histogram(a, bins = range(5))
# Below, hist indicates that there are 0 items in bin #0, 2 in bin #1, 4 in bin #3, 1 in bin #4.

print(hist)
# array([0, 2, 4, 1])   
#bin_edges indicates that bin #0 is the interval [0,1), bin #1 is [1,2), ..., bin #3 is [3,4).

print (bin_edges)
# array([0, 1, 2, 3, 4]))  
# Play with the above code, change the input to np.histogram and see how it works.

# But a picture is worth a thousand words:

import matplotlib.pyplot as plt
#plt.bar(bin_edges[:-1], hist, width = 1)
plt.plot(hist, color="red")
#plt.xlim(min(bin_edges), max(bin_edges))
plt.hist(a, bins = range(5))
plt.show() 