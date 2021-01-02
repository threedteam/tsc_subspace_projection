''''
Visually Induced Motion Sickness Dataset
Benchmark method of LinearDiscriminantAnalysis
Code Author: Xi Chen
Created on : 2020/12/05
'''
import pandas as pd
from Algorithm.MyAlgorithm import MyAlgorithm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject2.csv',header=None)
dataT = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject3.csv',header=None)
batchS = dataS.iloc[:,1:-3].values
batchS_label = dataS.iloc[:,-2].values
batchT = dataT.iloc[:,1:-3].values
batchT_label = dataT.iloc[:,-2].values

lda = LinearDiscriminantAnalysis(n_components=1)
Xs = lda.fit_transform(batchS,batchS_label)
Xt = lda.transform(batchT)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xs,batchS_label)
pre = knn.predict(Xt)
acc = accuracy_score(pre,batchT_label)
print(acc)
