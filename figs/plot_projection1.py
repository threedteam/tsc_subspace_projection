import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Algorithm.core import datapreprocessing
from Algorithm.MyAlgorithm import MyAlgorithm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

##5-6

dataS3 = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject3.csv',header=None)
dataT4 = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject4.csv',header=None)
batchS3 = dataS3.iloc[:,1:-3].values
batchS3_label = dataS3.iloc[:,-2].values
batchT4 = dataT4.iloc[:,1:-3].values
batchT4_label = dataT4.iloc[:,-2].values
Ms3 = np.mean(batchS3,0)
Mt4 = np.mean(batchT4,0)
vector = MyAlgorithm(Ms3,Mt4,batchS3,batchT4,batchS3_label,0.001,0.001,0.0001,1)
P = vector[:,:5]
Xs3 = np.dot(batchS3,P)
Xt4 = np.dot(batchT4,P)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xs3,batchS3_label)
pre = knn.predict(Xt4)
acc = accuracy_score(pre,batchT4_label)
print('Subject3-4',acc)

#############################################################################################################
dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject5.csv',header=None)
dataT = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\subject6.csv',header=None)
batchS = dataS.iloc[:,1:-3].values
batchS_label = dataS.iloc[:,-2].values
batchT = dataT.iloc[:,1:-3].values
batchT_label = dataT.iloc[:,-2].values
Ms = np.mean(batchS,0)
Mt = np.mean(batchT,0)
vector = MyAlgorithm(Ms,Mt,batchS,batchT,batchS_label,0.0001,0.0001,0.0001,0.1)
P = vector[:,:13]
Xs = np.dot(batchS,P)
Xt = np.dot(batchT,P)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xs,batchS_label)
pre = knn.predict(Xt)
acc = accuracy_score(pre,batchT_label)
print('Subject5-6',acc)


plt.figure(figsize=(12,18))
plt.subplots_adjust(wspace = 0.28)
plt.subplots_adjust(hspace=0.35)

#######################################################################################################################
plt.subplot(2,2,1)
dataS1 = Xs3
labelS1 = batchS3_label
index_1 = np.where(labelS1==0)
plt.scatter(dataS1[index_1,0],dataS1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(labelS1==1)
plt.scatter(dataS1[index_2,0],dataS1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
plt.xlabel('PC1',fontsize=20)
plt.ylabel('PC2',fontsize=20)
plt.title('Subject3',fontsize=20)

plt.subplot(2,2,2)
dataS1 = Xs
labelS1 = batchS_label
index_1 = np.where(labelS1==0)
plt.scatter(dataS1[index_1,0],dataS1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(labelS1==1)
plt.scatter(dataS1[index_2,0],dataS1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
plt.xlabel('PC1',fontsize=20)
plt.ylabel('PC2',fontsize=20)
plt.title('Subject5',fontsize=20)
plt.legend(bbox_to_anchor=(1.3,1.04),loc = 'upper right',fontsize=18)

plt.subplot(2,2,3)
dataS1 = Xt4
labelS1 = batchT4_label
index_1 = np.where(labelS1==0)
plt.scatter(dataS1[index_1,0],dataS1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(labelS1==1)
plt.scatter(dataS1[index_2,0],dataS1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
plt.xlabel('PC1',fontsize=20)
plt.ylabel('PC2',fontsize=20)
plt.title('Subject4',fontsize=20)

plt.subplot(2,2,4)
dataS1 = Xt
labelS1 = batchT_label
index_1 = np.where(labelS1==0)
plt.scatter(dataS1[index_1,0],dataS1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(labelS1==1)
plt.scatter(dataS1[index_2,0],dataS1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
plt.xlabel('PC1',fontsize=20)
plt.ylabel('PC2',fontsize=20)
plt.title('Subject6',fontsize=20)




# plt.subplot(2,2,3)
# dataS1 = Xs
# labelS1 = batchS_label
# index_1 = np.where(labelS1==0)
# plt.scatter(dataS1[index_1,0],dataS1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
# index_2 =np.where(labelS1==1)
# plt.scatter(dataS1[index_2,0],dataS1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
# plt.xlabel('PC1',fontsize=20)
# plt.ylabel('PC2',fontsize=20)
# plt.title('Subject',fontsize=20)
#
# plt.subplot(2,2,4)
# dataS1 = Xt
# labelS1 = batchT_label
# index_1 = np.where(labelS1==0)
# plt.scatter(dataS1[index_1,0],dataS1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
# index_2 =np.where(labelS1==1)
# plt.scatter(dataS1[index_2,0],dataS1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
# plt.xlabel('PC1',fontsize=20)
# plt.ylabel('PC2',fontsize=20)
# plt.title('Subject2',fontsize=20)

plt.show()