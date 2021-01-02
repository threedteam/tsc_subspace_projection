'''
Public sensors drift dataset
Benchmark method of PCA
Code Author: Xi Chen
Created on : 2020/12/01
'''

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from Algorithm.core import datapreprocessing
dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\batch5.csv')
dataT = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch6.csv")
batchS,batchS_label = datapreprocessing(dataS)
batchT,batchT_label = datapreprocessing(dataT)

for i in range(1,128):
    pca = PCA(n_components=i)
    Xs = pca.fit_transform(batchS)
    Xt = pca.transform(batchT)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Xs,batchS_label)
    pre = knn.predict(Xt)
    acc = accuracy_score(pre,batchT_label)
    print('d:{},acc:{}'.format(i,acc))

