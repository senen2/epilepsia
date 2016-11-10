from scipy import signal as sg
import numpy as np
print sg.convolve([[255, 7, 3],
                   [212, 240, 4],
                   [218, 216, 230]], [[1, -1]], "valid")

img = np.array([[255, 7, 3], [212, 240, 4], [218, 216, 230]])
c =  np.array([[1, -1]])
print sg.convolve(img, c, "valid")


# gives
# [[-248   -4]
#  [  28 -236]
#  [  -2   14]]'''

