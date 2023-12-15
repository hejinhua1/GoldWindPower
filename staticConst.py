import numpy as np

class AtmosphericData:
    def __init__(self, station=1):
        self.station = station
        if station == 0:
            self.shape = (70080, 8)
            self.data_max = np.array([1.04041700e+03, 9.97630000e+01, 3.28256527e+01, 2.88263586e+01,
                                      3.59998000e+02, 2.87637742e+01, 2.07999992e+01, 1.52229969e+05])
            self.data_min = np.array([ 9.90755000e+02,  2.97220000e+01, -4.40000000e+00, -1.51710000e+01,
                                       0.00000000e+00,  0.00000000e+00,  3.00000012e-01,  0.00000000e+00])
            self.data_mean = np.array([1.01646744e+03, 7.88497561e+01, 1.66942704e+01, 1.28827215e+01,
                                       1.43536532e+02, 7.18600882e+00, 6.47229166e+00, 5.15366037e+04])
            self.data_var = np.array([8.72525947e+01, 1.56385866e+02, 6.72709466e+01, 8.78045320e+01,
                                      9.86146174e+03, 1.08993525e+01, 9.88437347e+00, 2.41690097e+09])
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
        elif station == 1:
            self.shape = (70080, 8)
            self.data_max = np.array([9.87746652e+02,1.00000000e+02, 3.35194468e+01, 2.72610000e+01,
                                      3.60000000e+02, 1.52400000e+01, 1.83999996e+01, 8.45808438e+04])
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
        elif station == 2:
            self.shape = (70080, 8)
            self.data_max = np.array([1.02485600e+03, 9.95230000e+01, 3.56930000e+01, 2.71285585e+01,
                                      3.60000000e+02, 1.92710000e+01, 3.09799995e+01, 4.01168008e+04])
            self.data_min = np.array([982.661, 37.7155222, 3.016, -7.88178312,
                                      0., 0., 0., 0.])
            self.data_mean = np.array([1.00484308e+03, 8.13231955e+01, 2.06276171e+01, 1.71761209e+01,
                                       9.65032060e+01, 7.40415078e+00, 7.70322373e+00, 1.79627152e+04])
            self.data_var = np.array([4.81573849e+01, 1.20702222e+02, 4.00721349e+01, 4.54939877e+01,
                                      9.31235293e+03, 8.71665593e+00, 1.19447926e+01, 1.68375174e+08])
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
        elif station == 3:
            self.shape = (70080, 8)
            self.data_max = np.array([1.03706900e+03, 9.97650000e+01, 3.75689086e+01, 2.85410000e+01,
                                      3.60000000e+02, 3.02270000e+01, 2.13099995e+01, 7.56735938e+04])
            self.data_min = np.array([9.65260000e+02,  2.48430000e+01, -4.87900000e+00, -1.54050000e+01,
                                      0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
            self.data_mean = np.array([1.01384546e+03, 8.01720343e+01, 1.82279228e+01, 1.46075517e+01,
                                       1.57978440e+02, 5.65139873e+00, 5.27528534e+00, 1.56773756e+04])
            self.data_var = np.array([7.99070123e+01, 1.59712241e+02, 6.08266157e+01, 7.70752990e+01,
                                      1.25729945e+04, 8.07680592e+00, 6.26966305e+00, 2.85430575e+08])
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
        else: # station==1
            self.shape = (70080, 8)
            self.data_max = np.array([9.87746652e+02,1.00000000e+02, 3.35194468e+01, 2.72610000e+01,
                                      3.60000000e+02, 1.52400000e+01, 1.83999996e+01, 8.45808438e+04])
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