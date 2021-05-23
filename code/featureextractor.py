import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Read all files in the path file
def eachFile(path):
    '''
    File path read
    :param path:Data folder path
    :return: Data path, data file name
    '''
    file_list = []
    pathdir = os.listdir(path)
    for allDir in pathdir:
        child = os.path.join('%s%s'%(path,allDir))
        file_list.append(child)
    return file_list,pathdir

def getLabel(pathdir):
    '''
    obtain label
    :param pathdir:file name
    :return:
    '''
    label = pathdir.split('_')[1]
    if label == "GCO":
        return 1
    if label == "GEa":
        return 2
    if label == "GEy":
        return 3
    if label == "GMe":
        return 4


# filter
def filter(path):
    standard_data = pd.read_csv(path,header=None,sep="\t")
    sensor_data = standard_data.iloc[:, 1:]
    sensor_data1 = np.array(sensor_data)
    sensor_data2 = sensor_data1.copy()
    t = 5
    m = len(sensor_data1)
    for j in range(t - 1, m):
        for k in range(8):
            sum = np.sum(sensor_data1[j - t + 1:j + 1, k])  ##sum
            min = np.amin(sensor_data1[j - t + 1:j + 1, k])  # minimum value
            max = np.amax(sensor_data1[j - t + 1:j + 1, k])  # maximum value
            sensor_data2[j, k] = (sum - max - min) / (t - 2)  # filtered value
    sensor_data2 = pd.DataFrame(sensor_data2)
    return sensor_data2

def downsample(path,name):
    filter_file_path = "F:/Machine_Learning/Transductive_Transfer_Learning/板间数据/dataset/FB5/"
    data = filter(path)
    result = pd.DataFrame()
    for j in range(8):
        i = 1
        m = []
        data1 = data.iloc[:,j]
        while i*100 < len(data1):
            mdata =  data1.iloc[(i-1)*100:i*100]
            mvalue = np.mean(mdata)
            m.append(mvalue)
            i = i + 1
        n = pd.Series(m)
        result = pd.concat([result,n],axis=1)
    result.to_csv(filter_file_path + name, sep='\t', index=False, header=False)

def featureextractor(file_list,pathdir):
    """特征提取"""
    standard_data = pd.read_csv(file_list,header=None,sep="\t") ## Read the preprocessed data
    standard_data = standard_data.values
    label = getLabel(pathdir)
    feature = []
    for i in range(8):
        baseline = np.mean(standard_data[0:50, i])  ##  baseline phase, the baseline value is the mean value of the baseline phase
        injection_sort = standard_data[50:150, i] ## Sample stage
        steady = np.mean(injection_sort[-40:])             ## The mean value of the last 40 data points during the injection stage was taken as the steady state value
        steady_value = steady - baseline                   ## teady-state value subtracted from baseline
        feature.extend([steady_value])
    feature.extend([label])
    return feature






if __name__ == "__main__":
    file_list,pathdir = eachFile("F:/Machine_Learning/Transductive_Transfer_Learning/板间数据/dataset/FB5/")
    data = pd.read_csv(file_list[3], header=None, sep="\t")
    x = [i for i in range(len(data))]
    plt.plot(x,data.iloc[:,2])


