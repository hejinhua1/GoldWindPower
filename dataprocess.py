import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# 读取CSV文件
forecast_file_path = 'E:/HJHCloud/Seafile/私人资料库/金风科技比赛/决赛数据/台风/预测/815_forecast.csv'
real_file_path = 'E:/HJHCloud/Seafile/私人资料库/金风科技比赛/决赛数据/台风/实测/815.csv'
forecast_data = pd.read_csv(forecast_file_path)
real_data = pd.read_csv(real_file_path)
forecast_data = forecast_data.drop(forecast_data.columns[0], axis=1)
real_data = real_data.drop(real_data.columns[0], axis=1)
# 合并两个DataFrame
merged_df = pd.merge(forecast_data, real_data, on='dtime', how='outer')
feature_df = merged_df.drop(merged_df.columns[0], axis=1)
# 用均值填充缺失值
feature_df = feature_df.fillna(feature_df.mean())
# 提取特征数据

features = feature_df.values

# 使用MinMaxScaler进行归一化处理
minmaxscaler = MinMaxScaler()
standardscaler = StandardScaler()
minmax_data = minmaxscaler.fit_transform(features)
standard_data = standardscaler.fit_transform(features)


# 保存归一化后的数据为npy格式
save_path = 'E:/HJHCloud/Seafile/startup/GoldWindPower/data'
np.save(save_path + 'minmax_data.npy', minmax_data)
np.save(save_path + 'std_data.npy', standard_data)
# 保存merged_df为CSV文件
merged_df.to_csv(save_path + 'merged_data.csv', index=False)
