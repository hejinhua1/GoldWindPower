import numpy as np

class AtmosphericData:
    def __init__(self):
        self.shape = (70080, 8)
        self.data_max = np.array([9.87746652e+02,1.00000000e+02, 3.35194468e+01, 2.72610000e+01,
                                  3.60000000e+02, 1.52400000e+01, 4.38600006e+01, 8.45808438e+04])
        self.data_min = np.array([949.877, 21.834, 1.24, -12.17596966, 0., 0., 0., 0.])
        self.data_mean = np.array()
        self.data_std = np.array()

        self.z_std_dict = {}
        self.Typhoon = {

        }



if __name__ == '__main__':
    # 创建 AtmosphericData 实例
    atmospheric_data = AtmosphericData()

    # 现在你可以访问数据和字典
    print(atmospheric_data.data_max)
    print(atmospheric_data.z_std_dict['50'])
    print('finished')