'''
Created on Oct 31, 2016

@author: botpi
'''
import tensorflow as tf
import numpy as np
from apiepi import *
import epinn2

r = []
r.append(["File", "Class"])

for i in range(3):
    params = train("train_%s nz" % (i+1))
    r = r + evalue(params, test)
    