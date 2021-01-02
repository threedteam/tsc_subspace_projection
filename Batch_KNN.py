''''
Public sensors drift dataset
Benchmark method of KNeighborsClassifier
Code Author: Xi Chen
Created on : 2020/12/01
'''

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from Algorithm.core import datapreprocessing
from sklearn.metrics import accuracy_score

dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\batch9.csv')
dataT = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch10.csv")
batchS,batchS_label = datapreprocessing(dataS)
batchT,batchT_label = datapreprocessing(dataT)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(batchS,batchS_label)
pre = knn.predict(batchT)
acc = accuracy_score(pre,batchT_label)
print(acc)
