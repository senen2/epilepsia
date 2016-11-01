'''
Created on 14/09/2016

@author: botpi
'''
import numpy as np
import scipy.io
import os
from apiDB import DB

group = "train_1"
directory = "c:/concursos/epilepsia/%s/" % group

db = DB()
db.exe("truncate table pat1_1")
for file in os.listdir(directory):
    mat = scipy.io.loadmat(directory + file)
    print file
    data = mat['dataStruct'][0][0][0]
    name = file.split(".")[0]
    for t in range(240000):
        for channel in range(16):
            db.exe("insert into pat1_1 (IDchannel, time, val) values (%s, %s, %s)" % (channel, t, data[t][channel]))
        db.commit()

db.close()