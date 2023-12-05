import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
merged_df = merged_df.drop(merged_df.columns[0], axis=1)
# 用均值填充缺失值
filled_df = merged_df.fillna(merged_df.mean())
# 提取特征数据

features = filled_df.values

# 使用MinMaxScaler进行归一化处理
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(features)

# 保存归一化后的数据为npy格式
np.save('normalized_data.npy', normalized_data)
