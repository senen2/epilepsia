'''
Compute correlation coefficient and save file

Created on 14/09/2016

test 1 total 1584, nan 47, not nan 1537
test 2 total 2256 nan 29
test 3 total 2286 nan 4

test new 1 216 11 227

en los train los nan se ignoran porque afectan negativamente el entrenamiento ya que hay 0 y 1 en este grupo
en los test se asumen como 0 para completar el archivo de envio (pueden ser 1 porque evaluan lo mismo)

@author: botpi
'''
import numpy as np
import scipy.io, scipy.stats
import os
import matplotlib.pyplot as plt
import pandas as pd
from pyodbc import Row

print "begin"
directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/epilepsia/"

guide = pd.read_csv('train_and_test_data_labels_safe.csv') #.values
guide['patient'] = guide['image'].apply(lambda x: x.split('_')[0])
guide['dataset'] = guide['image'].apply(lambda x: 'train' if len(x.split('_'))==3 else 'test')

#z = np.diag(np.ones(16))
z = np.ones([16,16])
for pat in range(3):
    patient = str(pat+1)
    rows = guide.loc[(guide['patient'] == patient) & (guide['safe'] == 1)]
    t = {}
    nnan = 0
    pos = 0
    nfound = 0
    for index, row in rows.iterrows():
        filename = "%s%s_%s/%s" % (directory, row['dataset'], patient, row['image'])
        #print row['image']
        if os.path.isfile(filename):
            mat = scipy.io.loadmat(filename)
            data = mat['dataStruct'][0][0][0]
            name = row['image'].split(".")[0]
        
            b = np.sum(data, axis=1)
            c = data[b!=0]       
            
            t[name] = {}
            t[name]["corr"] = np.corrcoef(c, rowvar=0)
            t[name]["label"] = row['class']
            if row['class'] == 1:
                pos +=1
            if np.isnan(t[name]["corr"]).any():
                print name
                if row['dataset'] == 'test':
                    t[name]["corr"] = z
                    nnan+=1
        else:
            nfound +=1
    print "patient", "total", "nan", "pos", "not_found"
    print patient, len(t), nnan, pos, nfound
    scipy.io.savemat("train %s_new" % patient, t, do_compression=True)

print "end"
