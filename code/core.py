"""
Code Author: Xi Chen
Created on  2020/12/01
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

def ToOne(dataSet):
    xmin=dataSet.min(axis=0)
    xmax=dataSet.max(axis=0)
    std=(dataSet-xmin)/(xmax-xmin)
    return std

def datapreprocessing(data):
    '''
    data preprocessing
    :param data:
    :return:
    '''
    dataset = data.iloc[:,:-1]
    label = data.iloc[:,-1].values
    data_temp = dataset.values
    dataset = (dataset/np.sqrt(sum(data_temp*data_temp))).values
    return dataset,label

def within_class_scatter(data,label):
    '''within class scatter matrix'''
    labelset = set(label)
    dim = data.shape[1]
    row = data.shape[0]
    Sw = np.zeros((dim,dim))

    for i in labelset:
        pos = np.where(label == i)
        X = data[pos]
        possize = np.size(pos)
        mean = np.mean(X,0)
        mean = np.array([mean])
        S = np.dot((X-mean).T,(X-mean))
        Sw = Sw + (possize/row)*S           
    return Sw

def within_class_scatter_nonweight(data,label):
    '''within class scatter matrix'''
    labelset = set(label)
    dim = data.shape[1]
    row = data.shape[0]
    Sw = np.zeros((dim,dim))

    for i in labelset:
        pos = np.where(label == i)
        X = data[pos]
        possize = np.size(pos)
        mean = np.mean(X,0)
        mean = np.array([mean])
        S = np.dot((X-mean).T,(X-mean))
        Sw = Sw + S           
    return Sw

def between_class_scatter(data,label):
    '''between class scatter matrix'''
    labelset = set(label)
    dim = data.shape[1]
    row = data.shape[0]
    Sb = np.zeros((dim,dim))
    total_mean = np.mean(data,0)
    total_mean = np.array([total_mean])

    for i in labelset:
        pos = np.where(label == i)
        X = data[pos]
        possize = np.size(pos)
        mean = np.mean(X,0)
        mean = np.array([mean])
        S = np.dot((mean-total_mean).T,(mean-total_mean))
        Sb = Sb + (possize/row)*S
    return Sb

def target_domain_scatter(data):
    '''target domain scatter matrix'''
    row = data.shape[0]
    H = np.eye(row) - (1/row)*np.ones((row,row))
    St = np.dot(np.dot(data.T,H),data)
    return St

def source_domain_scatter(data):
    '''source domain scatter matrix'''
    row = data.shape[0]
    H = np.eye(row) - (1 / row) * np.ones((row, row))
    Ss = np.dot(np.dot(data.T,H),data)
    return Ss

def Labelencoder(label):
    '''
    onehot encoder
    :param label: label
    :return:
    '''
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(label)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def HSIC(data,label):
    '''
    Hilbert-Schmidt independence criterion
    :param data: data
    :param label: label
    :return:
    '''
    row = data.shape[0]
    H = np.eye(row) - (1 / row) * np.ones((row, row))
    Y = Labelencoder(label)
    Z = np.dot(np.dot(Y.T,H),data)
    hsic = np.dot(Z.T,Z)
    return hsic



