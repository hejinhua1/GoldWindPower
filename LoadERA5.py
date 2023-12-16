import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tool import Data_normalizer
from datetime import datetime


class WindDataset(Dataset):
    def __init__(self, flag='train', station=1, Norm_type='std', M=9, N=9):
        self.station = station
        if station == 0:
            if Norm_type == 'std':
                directory_path = "./data/std_data_station0.npy"
            else:
                directory_path = "./data/minmax_data_station0.npy"
        elif station == 1:
            if Norm_type == 'std':
                directory_path = "./data/std_data_station1.npy"
            else:
                directory_path = "./data/minmax_data_station1.npy"
        elif station == 2:
            if Norm_type == 'std':
                directory_path = "./data/std_data_station2.npy"
            else:
                directory_path = "./data/minmax_data_station2.npy"
        elif station == 3:
            if Norm_type == 'std':
                directory_path = "./data/std_data_station3.npy"
            else:
                directory_path = "./data/minmax_data_station3.npy"
        else:
            if Norm_type == 'std':
                directory_path = "./data/std_data_station1.npy"
            else:
                directory_path = "./data/minmax_data_station1.npy"

        self.data = []

        # 遍历目录下的所有文件
        year_start = datetime(2021, 1, 1)
        self.yearSet_dict = {'train': ('2021-01-01', '2022-09-01'),
                             'vali': ('2022-09-01', '2022-12-31'),
                             'test': ('2022-09-01', '2022-12-31')}
        start_time = datetime.strptime(self.yearSet_dict[flag][0], '%Y-%m-%d')
        end_time = datetime.strptime(self.yearSet_dict[flag][1], '%Y-%m-%d')
        base_index = (start_time - year_start).days * 96
        index_length = (end_time - start_time).days * 96

        data_all = np.load(directory_path)
        data = data_all[base_index:base_index+index_length, :]

        for i in range(data.shape[0] - M - N + 1):
            history_input_data = data[i:i + M, :]
            #站在当前时刻，预测未来16-24个时刻的风速
            #但是这里输入了未来16个时刻的实际风速和实际功率，要把这16个时刻的风速和功率置零
            # history_input_data[-16:, 6:] = 0
            forecast_input_data = data[i + M:i + M + N, :6]
            output_data = data[i + M:i + M + N, :]
            self.data.append((history_input_data, forecast_input_data, output_data))
        self.M = M
        self.N = N
        self.Norm_type = Norm_type
        self.normalizer = Data_normalizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        history_input_data, forecast_input_data, output_data = self.data[idx]
        return history_input_data, forecast_input_data, output_data






if __name__ == '__main__':
    batch_size = 6400
    # 获取当前时间
    start_time = datetime.now()
    # 打印当前时间
    print("Current time:", start_time)


    wind_dataset = WindDataset(flag='train', Norm_type='maxmin', M=24, N=24)
    dataloader = DataLoader(wind_dataset, batch_size=batch_size, shuffle=True)

    # 获取当前时间
    current_time = datetime.now()
    # 计算dataloader的时间
    delt_time = current_time - start_time
    # 打印当前时间
    print("Current time:", delt_time)
    i = 0
    for (hx, fx, y) in dataloader:
        print(hx.shape)
        print(fx.shape)
        print(y.shape)
        i = i+1
        print(i)
