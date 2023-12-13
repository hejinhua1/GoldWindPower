import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tool import Data_normalizer
import datetime


class WindDataset(Dataset):
    def __init__(self, flag='train', Year=None, Norm_type='std', M=9, N=9):
        if Norm_type == 'std':
            directory_path = "/home/hjh/data/std data/"
        else:
            directory_path = "/home/hjh/data/minmax data/"
        self.data = []

        # 遍历目录下的所有文件
        self.yearSet_dict = {'train': list(range(1979, 2018)),
                             'vali': [2019, 2022],
                             'test': [2018, 2020, 2021]}
        if Year is None:
            # 获取目录下所有年份，按照年份排序
            year_paths = sorted(os.listdir(directory_path))

            for year_path in year_paths:
                if int(year_path[4:8]) not in self.yearSet_dict[flag]:
                    continue
                data_path = os.path.join(directory_path, year_path)
                data_year = np.load(data_path)
                self.data.append(data_year)
        else:
            data_path = os.path.join(directory_path, 'data' + Year + '.npy')
            data_year = np.load(data_path)
            self.data.append(data_year)

        self.data = np.concatenate(self.data, axis=0)
        self.M = M
        self.N = N
        self.Norm_type = Norm_type
        self.normalizer = Data_normalizer()

    def __len__(self):
        return self.data.shape[0] - self.M - self.N + 1

    def __getitem__(self, idx):
        if idx <= self.data.shape[0] - self.N - self.M:
            input_data, output_data = self.data[idx:idx+self.M], self.data[idx+self.M:idx+self.M+self.N]
            # if self.Norm_type == 'maxmin':
            #     input_data = self.normalizer.normalize_min(input_data)
            #     output_data = self.normalizer.normalize_min(output_data)
            #
            # else:
            #     input_data = self.normalizer.normalize_std(input_data)
            #     output_data = self.normalizer.normalize_std(output_data)
        return input_data, output_data






if __name__ == '__main__':
    batch_size = 200
    # 获取当前时间
    start_time = datetime.datetime.now()
    # 打印当前时间
    print("Current time:", start_time)


    wind_dataset = WindDataset(flag='train', Norm_type='maxmin', M=4, N=24)
    # dataloader = DataLoader(wind_dataset, batch_size=batch_size, shuffle=True)
    #
    # # 获取当前时间
    # current_time = datetime.datetime.now()
    # # 计算dataloader的时间
    # delt_time = current_time - start_time
    # # 打印当前时间
    # print("Current time:", delt_time)
    # i = 0
    # for x,y in dataloader:
    #     print(x.shape)
    #     print(y.shape)
    #     i = i+1
    #     print(i)
