#-*- coding:utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[map(float,line) for line in stringArr]
    #print datArr
    return mat(datArr)

def pca(dataMat,topNfeat=9999999):
    meanVals=mean(dataMat,axis=0)  #求均值
    meanRemoved=dataMat-meanVals   #减去均值
    covMat=cov(meanRemoved,rowvar=0) #协方差矩阵，rowvar=0表示一行代表一个样本
    eigVals,eigVects=linalg.eig(mat(covMat)) #求特征值和特征向量
    eigValIndice=argsort(eigVals) #对特征值从小到大排序
    eigValIndice=eigValIndice[:-(topNfeat+1):-1] #减去不要的维数
    redEigVects=eigVects[:,eigValIndice] #最大的n个特征值对应的特征向量
    lowDDataMat=meanRemoved*redEigVects #低维特征空间的数据
    reconMat=(lowDDataMat*redEigVects.T)+meanVals #重构数据
    #print reconMat
    return lowDDataMat,reconMat


def plotBestFit(dataSet1, dataSet2):
    dataArr1 = array(dataSet1)
    dataArr2 = array(dataSet2)
    n = shape(dataArr1)[0]
    n1 = shape(dataArr2)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    xcord3 = []
    ycord3 = []
    j = 0
    for i in range(n):
        xcord1.append(dataArr1[i, 0]);
        ycord1.append(dataArr1[i, 1])
        xcord2.append(dataArr2[i, 0]);
        ycord2.append(dataArr2[i, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    mata = loadDataSet('data/textSet.txt')
    a, b = pca(mata, 2)
    plotBestFit(a, b)
