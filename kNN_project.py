#-*- coding:utf-8 –*-

from numpy import *
import operator
import pandas as pd

def classify0(inX, dataSet, labels, k):
    '''

    :param inX: 用于分类的输入向量inX
    :param dataSet: 输入的训练样本集
    :param labels: 标签向量（元素数目等于矩阵dataSet的行数）
    :param k: 用于选择最近邻的数目
    :return:
    '''

    # 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # tile()将inX行重复，扩充成dataSet的大小。 diffMat是矩阵相减结果
    sqDiffMat = diffMat**2                          # diffMat每个元素平方
    sqDistances = sqDiffMat.sum(axis=1)             # 按行求和，输出是1维ndarray
    distances = sqDistances**0.5                    # 开方

    # 选择距离最小的k个点
    sortedDistIndicies = distances.argsort()        # 排序：.argsort()返回数组值从小到大的索引值
                                                    # 如：a = [2,4,1,3]， a.argsort() 为 [3,1,4,2]
    classCount={}                                   # 创建字典classCount
    for i in range(k):                              # k: kNN的k
        voteIlabel = labels[sortedDistIndicies[i]]  # 排名第i的label赋给voteIlabel
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  # 为字典的键赋值: 统计标签I出现的次数
                                                                   # dict.get(key, default=None)
    # 排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
                                                    # sorted(iterable[, cmp[, key[, reverse]]]) 返回排序好的列表
                                                    # classCount.iteritems()返回字典的可迭代器（变成list形式）
                                                    # key=operator.itemgetter(1)表示根据第1（2）个值排序：标签出现的次数

    return sortedClassCount[0][0]                   # 返回排序好的列表中排名第一的键值对的键（排第一的标签）

# dataset: group是n行矩阵，每行表示一个样本
#          labels是n元列表，每个元素是对应对应索引的样本的标签

def file2matrix(filename):
    pdfile = pd.read_csv(filename)
    fileMat = pdfile.values
    fileMatLines  = fileMat.tolist()        # 读取行
    numberOfLines = fileMat.shape[0]        # 读取行数,从0开始计数（自动忽略了第一行标签行）
    returnMat = zeros((numberOfLines, 4))   # 创建返回的矩阵：行数为样本数，列数为特征数
    classLabelVector = []                   # 创建返回的list：对应矩阵样本的标签
    index = 0                               # 标签的索引

    for line in fileMatLines:
        returnMat[index, :] = line[1: 5]    # 给矩阵的第index行赋值：第index个样本的特征
        classLabelVector.append(line[0])    # 给标签list赋值
        index += 1                          # 每对一line进行操作，更新index

    return returnMat, classLabelVector

# 问题：numpy.ndarray()中必须为相同类型的变量。目前：将非数特征直接删掉，特征数从6变成4

def autoNorm(dataSet):
    minVals = dataSet.min(0)   # minVals中为每列的最小值（参数0表示从列中选取,提取成为一行） 返回一维ndarray
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals # ranges是一行矩阵，每个特征的range
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]       # array.shape[0] 行数
    normDataSet = dataSet - tile(minVals, (m,1)) # 矩阵每个元素相减。numpy中tile()函数将变量内容复制成输入矩阵同样大小的矩阵
                                                 # 将行矩阵minVals复制至m行，1列
    normDataSet = normDataSet/tile(ranges, (m,1))# 矩阵中的每个数相除
    return normDataSet, ranges, minVals

def tripClassTest():
    hoRatio = 0.10          # 测试集占比
    tripDataMat, tripLabels = file2matrix('train.csv')  # 数据格式转换
    normMat, ranges, minVals, = autoNorm(tripDataMat)
    m = normMat.shape[0]    # array.shape[0] 行数 ：总的样本个数
    numTestVecs = int(m * hoRatio)  # 测试向量的个数
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m,: ], tripLabels[numTestVecs: m], 3)
        '''
        classify0的参数：
        :param inX: 用于分类的输入向量inX
        :param dataSet: 输入的训练样本集
        :param labels: 标签向量（元素数目等于矩阵dataSet的行数）
        :param k: 用于选择最近邻的数目
        :return:
        '''
        # mat[i,:] 为矩阵第i行
        #
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult, tripLabels[i])
        if (classifierResult != tripLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" %(errorCount/float(numTestVecs))