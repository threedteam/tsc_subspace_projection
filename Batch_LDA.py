'''
Public sensors drift dataset
Benchmark method of LinearDiscriminantAnalysis
Code Author: Xi Chen
Created on : 2020/12/01
'''

import pandas as pd
from Algorithm.core import datapreprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\batch6.csv')
dataT = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch7.csv")
batchS,batchS_label = datapreprocessing(dataS)
batchT,batchT_label = datapreprocessing(dataT)

for i in range(1,6):
    lda = LinearDiscriminantAnalysis(n_components=i)
    Xs = lda.fit_transform(batchS,batchS_label)
    Xt = lda.transform(batchT)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Xs,batchS_label)
    pre = knn.predict(Xt)
    acc = accuracy_score(pre,batchT_label)
    print("d:{},acc:{}".format(i,acc))
