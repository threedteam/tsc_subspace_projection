''''
Visually Induced Motion Sickness Dataset
test domo
Code Author: Xi Chen
Created on : 2020/12/05
'''

import pandas as pd
from Algorithm.MyAlgorithm import MyAlgorithm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

trade_off = np.array([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4])

dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject1.csv',header=None)
dataT = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject3.csv',header=None)
batchS = dataS.iloc[:,1:-3].values
batchS_label = dataS.iloc[:,-2].values
batchT = dataT.iloc[:,1:-3].values
batchT_label = dataT.iloc[:,-2].values
Ms = np.mean(batchS,0)
Mt = np.mean(batchT,0)

vector = MyAlgorithm(Ms,Mt,batchS,batchT,batchS_label,10,1000,0.01,100)
P = vector[:,:1]
Xs = np.dot(batchS,P)
Xt = np.dot(batchT,P)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xs,batchS_label)
pre = knn.predict(Xt)
acc = accuracy_score(pre,batchT_label)
print('The test accuracy is : {}'.format(acc))