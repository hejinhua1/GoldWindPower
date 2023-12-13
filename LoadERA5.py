import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tool import Data_normalizer
from datetime import datetime


class WindDataset(Dataset):
    def __init__(self, flag='train', TyphoonMode=False, Norm_type='std', M=9, N=9):
        if Norm_type == 'std':
            directory_path = "E:/HJHCloud/Seafile/startup/GoldWindPower/data/minmax_data.npy"
        else:
            directory_path = "E:/HJHCloud/Seafile/startup/GoldWindPower/data/std_data.npy"
        self.data = []

        # 遍历目录下的所有文件
        year_start = datetime(2021, 1, 1)
        self.yearSet_dict = {'train': ('2021-01-01', '2022-08-31'),
                             'vali': ('2022-09-01', '2022-12-31'),
                             'test': ('2022-09-01', '2022-12-31')}
        start_time = datetime.strptime(self.yearSet_dict[flag][0], '%Y-%m-%d')
        end_time = datetime.strptime(self.yearSet_dict[flag][1], '%Y-%m-%d')
        base_index = (start_time - year_start).days * 96
        index_length = (end_time - start_time).days * 96

        data_all = np.load(directory_path)
        data = data_all[base_index:base_index+index_length, :]

        for i in range(data.shape[0] - M - N + 1):
            input_data = data[i:i + M, :]
            output_data = data[i + M:i + M + N, :]
            self.data.append((input_data, output_data))
        self.M = M
        self.N = N
        self.Norm_type = Norm_type
        self.normalizer = Data_normalizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data, output_data = self.data[idx]
        return input_data, output_data






if __name__ == '__main__':
    batch_size = 200
    # 获取当前时间
    start_time = datetime.now()
    # 打印当前时间
    print("Current time:", start_time)


    wind_dataset = WindDataset(flag='train', Norm_type='maxmin', M=4, N=24)
    dataloader = DataLoader(wind_dataset, batch_size=batch_size, shuffle=True)

    # 获取当前时间
    current_time = datetime.now()
    # 计算dataloader的时间
    delt_time = current_time - start_time
    # 打印当前时间
    print("Current time:", delt_time)
    i = 0
    for x,y in dataloader:
        print(x.shape)
        print(y.shape)
        i = i+1
        print(i)
