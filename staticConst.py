import numpy as np

class AtmosphericData:
    def __init__(self):
        self.shape = (70080, 8)
        self.data_max = np.array([9.87746652e+02,1.00000000e+02, 3.35194468e+01, 2.72610000e+01,
                                  3.60000000e+02, 1.52400000e+01, 4.38600006e+01, 8.45808438e+04])
        self.data_min = np.array([949.877, 21.834, 1.24, -12.17596966, 0., 0., 0., 0.])
        self.data_mean = np.array([9.71854897e+02, 8.06169909e+01, 2.10091182e+01, 1.72234666e+01,
                                   1.22577295e+02, 3.62705178e+00, 5.02223864e+00, 2.25289376e+04])
        self.data_var = np.array([3.50831492e+01, 2.55231449e+02, 3.73100321e+01, 5.00286586e+01,
                                  6.44395647e+03, 3.34081774e+00, 7.78148638e+00, 5.36015184e+08])
        self.data_std = np.sqrt(self.data_var)

        self.max_dict = {'power': self.data_max[7], 'rwind': self.data_max[6], 'fwind': self.data_max[5]}
        self.min_dict = {'power': self.data_min[7], 'rwind': self.data_min[6], 'fwind': self.data_min[5]}
        self.mean_dict = {'power': self.data_mean[7], 'rwind': self.data_mean[6], 'fwind': self.data_mean[5]}
        self.std_dict = {'power': self.data_std[7], 'rwind': self.data_std[6], 'fwind': self.data_std[5]}

        self.Typhoon = {
            '2021': [
                ('2021-06-03', '2021-06-05'),
                ('2021-08-04', '2021-08-07'),
                ('2021-09-09', '2021-09-12'),
                ('2021-10-11', '2021-10-12')
            ],
            '2022': [
                ('2022-06-03', '2022-06-05'),
                ('2022-08-04', '2022-08-07'),
                ('2022-09-09', '2022-09-12'),
                ('2022-10-11', '2022-10-12')
            ]
        }



if __name__ == '__main__':
    # 创建 AtmosphericData 实例
    atmospheric_data = AtmosphericData()

    # 现在你可以访问数据和字典
    print(atmospheric_data.data_max)
    print('finished')