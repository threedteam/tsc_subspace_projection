''''
Visually Induced Motion Sickness Dataset
Benchmark method of KNeighborsClassifier
Code Author: Xi Chen
Created on : 2020/12/05
'''

import pandas as pd
from Algorithm.MyAlgorithm import MyAlgorithm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject7.csv',header=None)
dataT = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject8.csv',header=None)
batchS = dataS.iloc[:,1:-3].values
batchS_label = dataS.iloc[:,-2].values
batchT = dataT.iloc[:,1:-3].values
batchT_label = dataT.iloc[:,-2].values

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(batchS,batchS_label)
pre = knn.predict(batchT)
acc = accuracy_score(pre,batchT_label)
print(acc)