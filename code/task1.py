import numpy as np
from scipy.io import loadmat

def load_data():
    '''
    fea:特征矩阵，大小为400×1024,每一行是一个样本
    gnd:400×1的列向量,每一行的数字表示当前这个样本的类别，一共40个类别
    '''
    data=loadmat('../ORL_32x32.mat')
    fea= data['fea'].astype(np.float32)
    gnd= data['gnd'].astype(np.float32)
    return data,fea,gnd

def dataSplit(fea, gnd, ntr=6):
    '''
    将数据随机划分为训练集和测试集，其中每个类别的训练样本数为 ntr，其余作为测试集。
    ntr: 每个类别的训练样本数
    '''
    # 获取总样本数和特征维度
    num_samples, feature_dim = fea.shape
    # 获取唯一类别
    unique_classes = np.unique(gnd)

    trainData = []
    trainGnd = []
    testData = []
    testGnd = []
    
    for cls in unique_classes:
        # 获取当前类别的样本索引
        cls_indices = np.where(gnd == cls)[0]
        np.random.shuffle(cls_indices)  # 随机打乱
        
        # 分割为训练集和测试集
        train_indices = cls_indices[:ntr]
        test_indices = cls_indices[ntr:]
        
        # 添加到训练集和测试集
        trainData.append(fea[train_indices])
        trainGnd.append(gnd[train_indices])
        testData.append(fea[test_indices])
        testGnd.append(gnd[test_indices])
    
    # 合并每个类别的样本，按照行方向进行拼接
    trainData = np.vstack(trainData)
    trainGnd = np.vstack(trainGnd)
    testData = np.vstack(testData)
    testGnd = np.vstack(testGnd)
    
    return trainData, trainGnd, testData, testGnd
