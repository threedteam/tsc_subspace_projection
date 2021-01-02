"""
Public sensors drift dataset
test demo
Code author: Xi Chen
Created on  2020/12/02
"""
import numpy as np
import pandas as pd
# from Algorithm.DMDMR import MyAlgorithm
from Algorithm.core import datapreprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from Algorithm.MyAlgorithm import MyAlgorithm

dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\batch1.csv')
dataT = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch6.csv")
batchS,batchS_label = datapreprocessing(dataS)
batchT,batchT_label = datapreprocessing(dataT)
Ms = np.mean(batchS,0)
Mt = np.mean(batchT,0)

vector = MyAlgorithm(Ms,Mt,batchS,batchT,batchS_label,0.1,10000,0.0001,0.0001)
P = vector[:,:8]
Xs = np.dot(batchS,P)
Xt = np.dot(batchT,P)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xs,batchS_label)
pre = knn.predict(Xt)
acc = accuracy_score(pre,batchT_label)
print('The test accuracy is : {}'.format(acc))
