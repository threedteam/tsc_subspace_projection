import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Algorithm.core import datapreprocessing
from Algorithm.MyAlgorithm import MyAlgorithm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
plt.rc('font',family='Times New Roman')



dataS = pd.read_csv(r'F:\Machine_Learning\Transductive_Transfer_Learning\batch1.csv')
########################################################################################################################
dataT2 = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch2.csv")
batchS,batchS_label = datapreprocessing(dataS)
batchT2,batchT2_label = datapreprocessing(dataT2)
Ms = np.mean(batchS,0)
Mt2 = np.mean(batchT2,0)
vector = MyAlgorithm(Ms,Mt2,batchS,batchT2,batchS_label,0.1,1000,0.1,10)
P2 = vector[:,:32]
Xs2 = np.dot(batchS,P2)
Xt2 = np.dot(batchT2,P2)
knn2 = KNeighborsClassifier(n_neighbors=1)
knn2.fit(Xs2,batchS_label)
pre2 = knn2.predict(Xt2)
acc = accuracy_score(pre2,batchT2_label)
print('batch1-2 accuracy:',acc)
##################################################################################################################

dataT5 = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch5.csv")
batchT5,batchT5_label = datapreprocessing(dataT5)
Ms = np.mean(batchS,0)
Mt5 = np.mean(batchT5,0)
vector = MyAlgorithm(Ms,Mt5,batchS,batchT5,batchS_label,1,10000,0.0001,1000)
P5 = vector[:,:76]
Xs5 = np.dot(batchS,P5)
Xt5 = np.dot(batchT5,P5)
knn5 = KNeighborsClassifier(n_neighbors=1)
knn5.fit(Xs5,batchS_label)
pre5 = knn5.predict(Xt5)
acc = accuracy_score(pre5,batchT5_label)
print('batch1-5 accuracy:',acc)
##################################################################################################################

dataT8 = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch7.csv")
batchT8,batchT8_label = datapreprocessing(dataT8 )
Ms = np.mean(batchS,0)
Mt8 = np.mean(batchT8,0)
vector = MyAlgorithm(Ms,Mt8,batchS,batchT8,batchS_label,10,100,0.001,100)
P8 = vector[:,:12]
Xs8 = np.dot(batchS,P8)
Xt8 = np.dot(batchT8,P8)
knn8 = KNeighborsClassifier(n_neighbors=1)
knn8.fit(Xs8,batchS_label)
pre8 = knn8.predict(Xt8)
acc = accuracy_score(pre8,batchT8_label)
print('batch1-7 accuracy:',acc)
#######################################################################################

dataT9 = pd.read_csv(r"F:\Machine_Learning\Transductive_Transfer_Learning\batch10.csv")
batchT9,batchT9_label = datapreprocessing(dataT9)
Ms = np.mean(batchS,0)
Mt9 = np.mean(batchT9,0)
vector = MyAlgorithm(Ms,Mt9,batchS,batchT9,batchS_label,0.0001,0.0001,0.0001,10)
P9 = vector[:,:114]
Xs9 = np.dot(batchS,P9)
Xt9 = np.dot(batchT9,P9)
knn9 = KNeighborsClassifier(n_neighbors=1)
knn9.fit(Xs9,batchS_label)
pre9 = knn9.predict(Xt9)
acc = accuracy_score(pre9,batchT9_label)
print('batch1-10 accuracy:',acc)

plt.figure(figsize=(12,18))
plt.subplots_adjust(wspace = 0.33)
plt.subplots_adjust(hspace=0.32)


plt.subplot(2,4,1)
data1 = Xs2
label1 = batchS_label
index_1 = np.where(label1==1)
plt.scatter(data1[index_1,0],data1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(label1==2)
plt.scatter(data1[index_2,0],data1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
index_3 =np.where(label1==3)
plt.scatter(data1[index_3,0],data1[index_3,1],marker='+',color = 'g',label = 'Class 3',s = 15)
index_4 =np.where(label1==4)
plt.scatter(data1[index_4,0],data1[index_4,1],marker='*',color = 'c',label = 'Class 4',s = 15)
index_5 =np.where(label1==5)
plt.scatter(data1[index_5,0],data1[index_5,1],marker='s',color = 'grey',label = 'Class 5',s = 15)
index_6 =np.where(label1==6)
plt.scatter(data1[index_6,0],data1[index_6,1],marker='1',color = 'y',label = 'Class 6',s = 15)
plt.xlabel('PC1',fontsize=21)
plt.ylabel('PC2',fontsize=21)
plt.title('Batch1',fontsize=21)
#
plt.subplot(2,4,2)
data1 = Xs5
label1 = batchS_label
index_1 = np.where(label1==1)
plt.scatter(data1[index_1,0],data1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(label1==2)
plt.scatter(data1[index_2,0],data1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
index_3 =np.where(label1==3)
plt.scatter(data1[index_3,0],data1[index_3,1],marker='+',color = 'g',label = 'Class 3',s = 15)
index_4 =np.where(label1==4)
plt.scatter(data1[index_4,0],data1[index_4,1],marker='*',color = 'c',label = 'Class 4',s = 15)
index_5 =np.where(label1==5)
plt.scatter(data1[index_5,0],data1[index_5,1],marker='s',color = 'grey',label = 'Class 5',s = 15)
index_6 =np.where(label1==6)
plt.scatter(data1[index_6,0],data1[index_6,1],marker='1',color = 'y',label = 'Class 6',s = 15)
plt.xlabel('PC1',fontsize=21)
plt.ylabel('PC2',fontsize=21)
plt.title('Batch1',fontsize=21)

plt.subplot(2,4,3)
data1 = Xs8
label1 = batchS_label
index_1 = np.where(label1==1)
plt.scatter(data1[index_1,0],data1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(label1==2)
plt.scatter(data1[index_2,0],data1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
index_3 =np.where(label1==3)
plt.scatter(data1[index_3,0],data1[index_3,1],marker='+',color = 'g',label = 'Class 3',s = 15)
index_4 =np.where(label1==4)
plt.scatter(data1[index_4,0],data1[index_4,1],marker='*',color = 'c',label = 'Class 4',s = 15)
index_5 =np.where(label1==5)
plt.scatter(data1[index_5,0],data1[index_5,1],marker='s',color = 'grey',label = 'Class 5',s = 15)
index_6 =np.where(label1==6)
plt.scatter(data1[index_6,0],data1[index_6,1],marker='1',color = 'y',label = 'Class 6',s = 15)
plt.xlabel('PC1',fontsize=21)
plt.ylabel('PC2',fontsize=21)
plt.title('Batch1',fontsize=21)

plt.subplot(2,4,4)
data1 = Xs9
label1 = batchS_label
index_1 = np.where(label1==1)
plt.scatter(data1[index_1,0],data1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(label1==2)
plt.scatter(data1[index_2,0],data1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
index_3 =np.where(label1==3)
plt.scatter(data1[index_3,0],data1[index_3,1],marker='+',color = 'g',label = 'Class 3',s = 15)
index_4 =np.where(label1==4)
plt.scatter(data1[index_4,0],data1[index_4,1],marker='*',color = 'c',label = 'Class 4',s = 15)
index_5 =np.where(label1==5)
plt.scatter(data1[index_5,0],data1[index_5,1],marker='s',color = 'grey',label = 'Class 5',s = 15)
index_6 =np.where(label1==6)
plt.scatter(data1[index_6,0],data1[index_6,1],marker='1',color = 'y',label = 'Class 6',s = 15)
plt.xlabel('PC1',fontsize=21)
plt.ylabel('PC2',fontsize=21)
plt.title('Batch1',fontsize=21)
plt.legend(bbox_to_anchor=(1.69,1.045),loc = 'upper right',fontsize=21)

plt.subplot(2,4,5)
data1 = Xt2
label1 = batchT2_label
index_1 = np.where(label1==1)
plt.scatter(data1[index_1,0],data1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(label1==2)
plt.scatter(data1[index_2,0],data1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
index_3 =np.where(label1==3)
plt.scatter(data1[index_3,0],data1[index_3,1],marker='+',color = 'g',label = 'Class 3',s = 15)
index_4 =np.where(label1==4)
plt.scatter(data1[index_4,0],data1[index_4,1],marker='*',color = 'c',label = 'Class 4',s = 15)
index_5 =np.where(label1==5)
plt.scatter(data1[index_5,0],data1[index_5,1],marker='s',color = 'grey',label = 'Class 5',s = 15)
index_6 =np.where(label1==6)
plt.scatter(data1[index_6,0],data1[index_6,1],marker='1',color = 'y',label = 'Class 6',s = 15)
plt.xlabel('PC1',fontsize=21)
plt.ylabel('PC2',fontsize=21)
plt.title('Batch2',fontsize=21)

plt.subplot(2,4,6)
data1 = Xt5
label1 = batchT5_label
index_1 = np.where(label1==1)
plt.scatter(data1[index_1,0],data1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(label1==2)
plt.scatter(data1[index_2,0],data1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
index_3 =np.where(label1==3)
plt.scatter(data1[index_3,0],data1[index_3,1],marker='+',color = 'g',label = 'Class 3',s = 15)
index_4 =np.where(label1==4)
plt.scatter(data1[index_4,0],data1[index_4,1],marker='*',color = 'c',label = 'Class 4',s = 15)
index_5 =np.where(label1==5)
plt.scatter(data1[index_5,0],data1[index_5,1],marker='s',color = 'grey',label = 'Class 5',s = 15)
# index_6 =np.where(label1==6)
# plt.scatter(data1[index_6,0],data1[index_6,1],marker='1',color = 'y',label = 'Class 6',s = 15)
plt.xlabel('PC1',fontsize=21)
plt.ylabel('PC2',fontsize=21)
plt.title('Batch5',fontsize=21)

plt.subplot(2,4,7)
data1 = Xt8
label1 = batchT8_label
index_1 = np.where(label1==1)
plt.scatter(data1[index_1,0],data1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(label1==2)
plt.scatter(data1[index_2,0],data1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
index_3 =np.where(label1==3)
plt.scatter(data1[index_3,0],data1[index_3,1],marker='+',color = 'g',label = 'Class 3',s = 15)
index_4 =np.where(label1==4)
plt.scatter(data1[index_4,0],data1[index_4,1],marker='*',color = 'c',label = 'Class 4',s = 15)
index_5 =np.where(label1==5)
plt.scatter(data1[index_5,0],data1[index_5,1],marker='s',color = 'grey',label = 'Class 5',s = 15)
index_6 =np.where(label1==6)
plt.scatter(data1[index_6,0],data1[index_6,1],marker='1',color = 'y',label = 'Class 6',s = 15)
plt.xlabel('PC1',fontsize=21)
plt.ylabel('PC2',fontsize=21)
plt.title('Batch7',fontsize=21)

plt.subplot(2,4,8)
data1 = Xt9
label1 = batchT9_label
index_1 = np.where(label1==1)
plt.scatter(data1[index_1,0],data1[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
index_2 =np.where(label1==2)
plt.scatter(data1[index_2,0],data1[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
index_3 =np.where(label1==3)
plt.scatter(data1[index_3,0],data1[index_3,1],marker='+',color = 'g',label = 'Class 3',s = 15)
index_4 =np.where(label1==4)
plt.scatter(data1[index_4,0],data1[index_4,1],marker='*',color = 'c',label = 'Class 4',s = 15)
index_5 =np.where(label1==5)
plt.scatter(data1[index_5,0],data1[index_5,1],marker='s',color = 'grey',label = 'Class 5',s = 15)
index_6 =np.where(label1==6)
plt.scatter(data1[index_6,0],data1[index_6,1],marker='1',color = 'y',label = 'Class 6',s = 15)
plt.xlabel('PC1',fontsize=21)
plt.ylabel('PC2',fontsize=21)
plt.title('Batch10',fontsize=21)
plt.show()


# data5 = Xs5
# label5 = batchS_label
# index_1 = np.where(label5==1)
# plt.scatter(data5[index_1,0],data5[index_1,1],marker='x',color = 'r',label = 'Class 1',s = 15)
# index_2 =np.where(label5==2)
# plt.scatter(data5[index_2,0],data5[index_2,1],marker='o',color = 'b',label = 'Class 2',s = 15)
# index_3 =np.where(label5==3)
# plt.scatter(data5[index_3,0],data5[index_3,1],marker='+',color = 'g',label = 'Class 3',s = 15)
# index_4 =np.where(label5==4)
# plt.scatter(data5[index_4,0],data5[index_4,1],marker='*',color = 'c',label = 'Class 4',s = 15)
# index_5 =np.where(label5==5)
# plt.scatter(data5[index_5,0],data5[index_5,1],marker='s',color = 'grey',label = 'Class 5',s = 15)
# plt.xlabel('PC1',fontsize=18)
# plt.ylabel('PC2',fontsize=18)
# plt.title('Batch2',fontsize=18)

# 选择1-2 1-5 1-6 1-8 1-9



