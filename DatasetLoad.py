import numpy as np
import gzip
import os
import torch
import platform
import pickle
from WHDY_vanilla_malicious_involved_fedavg.getData import *
import sys

"""

用于加载和处理数据集

"""


class DatasetLoad(object):
    def __init__(self, dataSetName, isIID):  # femnist 0
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0  # 初始化一个用于跟踪训练周期中索引的变量。

        if self.name == 'femnist':
            self.oarfDataSetConstruct(isIID)  # 构建数据集
        else:
            pass

    def oarfDataSetConstruct(self, isIID):
        """
        构建FEMNIST数据集 isIId 0
        """
        # 数据集文件路径
        data_dir = 'data/OARF'
        train_data_path = os.path.join(data_dir, 'FEMINIST.gz')
        train_labels_path = os.path.join(data_dir, 'CIFAR-10.gz')
        test_data_path = os.path.join(data_dir, 'Sent140.gz')
        test_labels_path = os.path.join(data_dir, 'Train_and_Test.gz')
        # 数据提取
        train_data = extract_data(train_data_path)  # (60000, 28, 28, 1)
        train_labels = extract_labels(train_labels_path)  # (60000, 10)
        test_data = extract_data(test_data_path)  # (10000, 28, 28, 1)
        test_labels = extract_labels(test_labels_path)  # (10000, 10)

        # CPU reduce size
        # train_data = train_data[:60]
        # train_labels = train_labels[:60]
        # test_data = test_data[:60]
        # test_labels = test_labels[:60]

        # 60000 data points
        assert train_data.shape[0] == train_labels.shape[0]
        assert test_data.shape[0] == test_labels.shape[0]

        self.train_data_size = train_data.shape[0]  # 60000
        self.test_data_size = test_data.shape[0]  # 10000
        # 确保数据的深度为1
        assert train_data.shape[3] == 1
        assert test_data.shape[3] == 1
        # 调整数据的形状，使其适合后续处理。
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        # 将数据类型转换为float32并归一化。
        train_data = train_data.astype(np.float32)
        train_data = np.multiply(train_data, 1.0 / 255.0)
        test_data = test_data.astype(np.float32)
        test_data = np.multiply(test_data, 1.0 / 255.0)

        if isIID:
            '''若为True，随机打乱训练数据和标签'''
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_data[order]
            self.train_label = train_labels[order]
        else:
            '''按标签排序'''
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_data[order]
            self.train_label = train_labels[order]

        self.test_data = test_data
        self.test_label = test_labels


def _read32(bytestream):
    """
    用于从字节流中读取32位无符号整数
    """
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


'''将文件名映射到数据集名称'''
database_name = {"FEMINIST.gz": "FEMINIST Dataset",
                 "CIFAR-10.gz": "CIFAR-10 Dataset",
                 "Sent140.gz": "Sent140 Dataset",
                 "Train_and_Test.gz": "Train and Test"}


def extract_data(filename):
    """
    用于从gzip文件中提取数据
    """
    """Extract the data into a 4D uint8 numpy array [index, y, x, depth]."""

    print('Extracting', database_name[filename.split('/')[-1]])
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)  # 魔法数字  第一位32位无符号整数 验证文件的格式是否正确
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in OARF data file: %s' %
                (magic, filename))
        # 每次调用都会从字节流中读取下一个32位无符号整数，所以每次调用的结果都是不同的。这些整数分别代表了文件中不同位置的特定元数据信息。
        num_data = _read32(bytestream)  # 图像的数量
        rows = _read32(bytestream)  # 每张图片的高度
        cols = _read32(bytestream)  # 每张图片的宽度
        buf = bytestream.read(rows * cols * num_data)  # 所有图像数据的总字节数，并从字节流中读取这些数据
        data = np.frombuffer(buf, dtype=np.uint8)  # 转换为numpy数组
        data = data.reshape(num_data, rows, cols, 1)  # 图像索引，图像高度，图像宽度，图像深度为1
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """
    Convert class labels from scalars to one-hot vectors.
    将类别的密集标签（dense labels）转换为独热编码 one-hot encoding
    labels_dense labels的一维numpy数组
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes  # 每个标签在独热编码中的起始索引偏移量
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[
        index_offset + labels_dense.ravel()] = 1  # ravel() 方法将 labels_dense 数组展平为一维数组,flat允许操作一维数组那样操作多维数组
    return labels_one_hot


def extract_labels(filename):
    """
    Extract the labels into a 1D uint8 numpy array [index].
    用于从gzip文件中提取标签数据
    """
    print('Extracting', database_name[filename.split('/')[-1]])
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)  # 读取第一个32位无符号整数，验证文件的格式是否正确
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in FEMNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)  # 整个标签的数量
        buf = bytestream.read(num_items)  # 从字节流中读取相应数量的字节数据
        labels = np.frombuffer(buf, dtype=np.uint8)  # 转换为numpy数组
        return dense_to_one_hot(labels)


if __name__ == "__main__":
    'test data set'
    oarfDataSet = GetDataSet('femnist', True)  # test NON-IID
    if type(oarfDataSet.train_data) is np.ndarray and type(oarfDataSet.test_data) is np.ndarray and \
            type(oarfDataSet.train_label) is np.ndarray and type(oarfDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(oarfDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(oarfDataSet.test_data.shape))
    print(oarfDataSet.train_label[0:100], oarfDataSet.train_label[11000:11100])


# Data Poisoning Attack
# add Gussian Noise to dataset

class AddGaussianNoise(object):
    """
    向数据中添加高斯噪声
    """

    def __init__(self, mean=0., std=1.):
        """
        高斯分布的均值和标准差
        """
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """
        call 允许类的实例像函数一样被调用
        将调整后的噪声添加到原始数据tensor上，并返回结果
        """
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        """
        返回一个字符串，格式为"AddGaussianNoise(mean=xx, std=yy)"，其中xx和yy分别是self.mean和self.std的值。
        """
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
