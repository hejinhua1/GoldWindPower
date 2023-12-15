import numpy as np
import matplotlib.pyplot as plt
from staticConst import AtmosphericData
import torch.nn.functional as F
import torch
from torch import nn

from PIL import Image
import os



def RSE(ypred, ytrue):
    """
    计算相对平方误差（Relative Squared Error）。

    参数：
    ypred (numpy.ndarray): 预测值的数组。
    ytrue (numpy.ndarray): 真实值的数组。

    返回：
    float: 相对平方误差的值。
    """
    rse = np.sqrt(np.square(ypred - ytrue).sum()) / \
            np.sqrt(np.square(ytrue - ytrue.mean()).sum())
    return rse

def MAE(ypred, ytrue):
    """
    计算平均绝对误差（Mean Absolute Error）。

    参数：
    ypred (numpy.ndarray): 预测值的数组。
    ytrue (numpy.ndarray): 真实值的数组。

    返回：
    float: 平均绝对误差的值。
    """
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel()
    mae = np.abs(ypred - ytrue).mean()
    return mae

def RMSE(ypred, ytrue):
    """
    计算均方根误差（Root Mean Squared Error）。

    参数：
    ypred (numpy.ndarray): 预测值的数组。
    ytrue (numpy.ndarray): 真实值的数组。

    返回：
    float: 均方根误差的值。
    """
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel()
    rmse = np.sqrt(np.square(ypred - ytrue).mean())
    return rmse

def ACC(ypred, ytrue, mean_observed):
    """
    计算异常相关系数（Anomaly Correlation Coefficient, ACC）

    参数：
    ypred (numpy.ndarray): 预测值的数组
    ytrue (numpy.ndarray): 真实值的数组
    mean_observed (float): 真实值的平均值

    返回：
    float: ACC
    """
    # 计算异常相关系数
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel()


    acc_numerator = np.sum((ypred - mean_observed) * (ytrue - mean_observed))
    acc_denominator_model = np.sqrt(np.sum((ypred - mean_observed) ** 2))
    acc_denominator_observed = np.sqrt(np.sum((ytrue - mean_observed) ** 2))

    acc = acc_numerator / (acc_denominator_model * acc_denominator_observed)

    return acc

def R2_score(ypred, ytrue):
    """
    计算R2_score

    参数：
    ypred (numpy.ndarray): 预测值的数组
    ytrue (numpy.ndarray): 真实值的数组

    返回：
    float: R2_score
    """
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel()
    r2_score = 1 - np.sum(np.square(ypred - ytrue)) / np.sum(np.square(ytrue - np.mean(ytrue)))
    return r2_score

def quantile_loss(ypred, ytrue, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    '''
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()

def SMAPE(ypred, ytrue):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel() + 1e-4
    mean_y = (ytrue + ypred) / 2.
    return np.mean(np.abs((ytrue - ypred) \
        / mean_y))

def MAPE(ypred, ytrue, ymax):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) \
        / ymax))

class Data_normalizer(AtmosphericData):
    def __init__(self, station=1):
        super(Data_normalizer, self).__init__(station=station)

    def normalize_std(self, data):
        '''
        data:(T,E,C,H,W)
        self.data_mean:(E,C)
        self.data_std:(E,C)
        '''
        # 这里可以根据需要进行标准化化
        out = np.zeros_like(data)
        for i in range(6):
            for j in range(13):
                out[:, i, j, :, :] = (data[:, i, j, :, :] - self.data_mean[i, j]) / self.data_std[i, j]
        return out

    def normalize_min(self, data):
        '''
        data:(T,E,C,H,W)
        self.data_mean:(E,C)
        self.data_std:(E,C)
        '''
        # 这里可以根据需要进行标准化化
        out = np.zeros_like(data)
        for i in range(6):
            for j in range(13):
                out[:, i, j, :, :] = (data[:, i, j, :, :] - self.data_min[i, j]) / (self.data_max[i, j] - self.data_min[i, j])
        return out

    def inverse_normalize_std(self, data):
        '''
        data:(B,T,E,C,H,W)
        data_mean:(E,C)
        data_std:(E,C)
        '''
        # 这里可以根据需要进行标准化化的逆过程
        out = np.zeros_like(data)
        for i in range(6):
            for j in range(13):
                out[:, :, i, j, :, :] = data[:, :, i, j, :, :] * self.data_std[i, j] + self.data_mean[i, j]
        return out

    def inverse_normalize_min(self, data):
        '''
        data:(B,T,E,C,H,W)
        data_mean:(E,C)
        data_std:(E,C)
        '''
        # 这里可以根据需要进行标准化化的逆过程
        out = np.zeros_like(data)
        for i in range(6):
            for j in range(13):
                out[:, :, i, j, :, :] = data[:, :, i, j, :, :] * (self.data_max[i, j] - self.data_min[i, j]) + self.data_min[i, j]
        return out

    def inverse_target(self, y, y_hat, target='p', Norm_type='std'):
        """
        用于将经过归一化后的数据反向还原成原始数据的函数。

        参数:
        y (numpy.ndarray): 包含目标变量（标签）的归一化数据。
        y_hat (numpy.ndarray): 包含模型预测结果的归一化数据。
        target (str): 目标变量（标签）的名称，可选'u'或'v'。

        返回:
        tuple: 包含两个numpy.ndarray的元组，第一个是还原后的目标变量（标签），第二个是还原后的模型预测结果。
        """
        if target == 'p':
            data_mean = self.mean_dict['power']
            data_std = self.std_dict['power']
            data_max = self.max_dict['power']
            data_min = self.min_dict['power']
        else:
            data_mean = self.mean_dict['rwind']
            data_std = self.std_dict['rwind']
            data_max = self.max_dict['rwind']
            data_min = self.min_dict['rwind']

        if Norm_type == 'std':
            # 如果选择标准化，使用均值和标准差来反向还原数据
            y_raw = y * data_std + data_mean
            y_hat_raw = y_hat * data_std + data_mean
        else:
            # 如果选择最小-最大归一化，使用最小值和最大值来反向还原数据
            y_raw = y * (data_max - data_min) + data_min
            y_hat_raw = y_hat * (data_max - data_min) + data_min

        return y_raw, y_hat_raw



class Weighted_loss(nn.Module):
    def __init__(self, loss_weight=(0.77, 0.54)):
        super(Weighted_loss,self).__init__()
        self.loss_weight = loss_weight

    def forward(self, u_loss, v_loss):
        loss = (self.loss_weight[0] * u_loss + self.loss_weight[1] * v_loss)/(self.loss_weight[0] + self.loss_weight[1])
        return loss

if __name__ == '__main__':
    # Create Data_normalizer instance with station=2
    data_normalizer = Data_normalizer(station=3)

    # Access data and dictionaries as needed
    print(data_normalizer.data_max)
    print('finished')