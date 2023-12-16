import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# # 读取CSV文件
# forecast_file_path = 'E:/HJHCloud/Seafile/私人资料库/金风科技比赛/决赛数据/台风/预测/815_forecast.csv'
# real_file_path = 'E:/HJHCloud/Seafile/私人资料库/金风科技比赛/决赛数据/台风/实测/1_815.csv'
# forecast_data = pd.read_csv(forecast_file_path)
# real_data = pd.read_csv(real_file_path)
# forecast_data = forecast_data.drop(forecast_data.columns[0], axis=1)
# real_data = real_data.drop(real_data.columns[0], axis=1)
# # 将时间列转换为datetime格式
# forecast_data['dtime'] = pd.to_datetime(forecast_data['dtime'])
# real_data['dtime'] = pd.to_datetime(real_data['dtime'])
# # 合并两个DataFrame
# merged_df = pd.merge(forecast_data, real_data, on='dtime', how='left')
# feature_df = merged_df.drop(merged_df.columns[0], axis=1)
# # 用均值填充缺失值
# feature_df = feature_df.fillna(feature_df.mean())
# # 提取特征数据
#
# features = feature_df.values
#
# # 使用MinMaxScaler进行归一化处理
# minmaxscaler = MinMaxScaler()
# standardscaler = StandardScaler()
# minmax_data = minmaxscaler.fit_transform(features)
# standard_data = standardscaler.fit_transform(features)
# print(minmaxscaler.data_max_)
# print(minmaxscaler.data_min_)
# print(standardscaler.mean_)
# print(standardscaler.var_)
#
# # 保存归一化后的数据为npy格式
# save_path = 'E:/HJHCloud/Seafile/startup/GoldWindPower/data/'
# np.save(save_path + 'minmax_data_station1.npy', minmax_data)
# np.save(save_path + 'std_data_station1.npy', standard_data)
# # 保存merged_df为CSV文件
# merged_df.to_csv(save_path + 'merged_data_station1.csv', index=False)
# print('Done!')

# save_path = 'E:/HJHCloud/Seafile/startup/GoldWindPower/data/'
# data20212022 = np.load(save_path + 'WindUTC20212022.npy')
# wind20212022 = np.sqrt(data20212022[:, 0] ** 2 + data20212022[:, 1] ** 2)
# merged_df = pd.read_csv(save_path + 'merged_data.csv')
# merged_df = merged_df.fillna(0)
# rw_col = merged_df['r_wspd'].values #r_apower, wspd_70
# rp_col = merged_df['r_apower'].values
# fw_col = merged_df['wspd_70'].values
# correlation1 = np.corrcoef(wind20212022[:-8], rw_col[8*4::4], rowvar=False)
# correlation2 = np.corrcoef(rw_col[8*4::4], rp_col[8*4::4], rowvar=False)
# correlation3 = np.corrcoef(wind20212022[:-8], rp_col[8*4::4], rowvar=False)
# correlation4 = np.corrcoef(wind20212022[:-8], fw_col[8*4::4], rowvar=False)
# correlation5 = np.corrcoef(fw_col[8*4::4], rp_col[8*4::4], rowvar=False)
# print(correlation1)

# station = 1
# minmax_path = './data/minmax_data_station{}.npy'.format(station)
# std_path = './data/std_data_station{}.npy'.format(station)
# minmaxdata = np.load(minmax_path)
# stddata = np.load(std_path)
# minmax_data_train = minmaxdata[:58368]
# minmax_data_test = minmaxdata[58368:]
# std_data_train = stddata[:58368]
# std_data_test = stddata[58368:]
# print(minmax_data_train.shape)
# print(minmax_data_test.shape)
# print(std_data_train.shape)
# print(std_data_test.shape)
# np.save('./data/minmax_data_station{}_train.npy'.format(station), minmax_data_train)
# np.save('./data/minmax_data_station{}_test.npy'.format(station), minmax_data_test)
# np.save('./data/std_data_station{}_train.npy'.format(station), std_data_train)
# np.save('./data/std_data_station{}_test.npy'.format(station), std_data_test)

station = 3
raw_path = './data/merged_data_station{}.csv'.format(station)
raw_data = pd.read_csv(raw_path)
raw_data = raw_data.drop(raw_data.columns[0], axis=1)
raw_data = raw_data.fillna(raw_data.mean())
data_wind = raw_data[['wspd_70','r_wspd']].values
missing_values = np.isnan(data_wind)

# Count the number of missing values for each column
num_missing_wspd_70 = np.sum(missing_values[:, 0])
num_missing_r_wspd = np.sum(missing_values[:, 1])

# Print the results
print(f'Number of missing values for wspd_70: {num_missing_wspd_70}')
print(f'Number of missing values for r_wspd: {num_missing_r_wspd}')
# Calculate Mean Absolute Error (MAE)
mae_wspd_70 = np.mean(np.abs(data_wind[:, 1] - data_wind[:, 0]))

# Calculate Root Mean Squared Error (RMSE)
rmse_wspd_70 = np.sqrt(np.mean((data_wind[:, 1] - data_wind[:, 0])**2))


# Print the results
print(f'MAE for wspd_70: {mae_wspd_70}')
print(f'RMSE for wspd_70: {rmse_wspd_70}')
