import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
real_file_path = '../data/datamerged_data.csv'
forecast_data = pd.read_csv(real_file_path)

# 提取'r_wspd'和'r_apower'列
r_wspd = forecast_data['wspd_70']
r_apower = forecast_data['r_wspd']

# 绘制散点图
plt.scatter(r_wspd, r_apower, alpha=0.5)  # alpha用于设置点的透明度，0为完全透明，1为完全不透明
plt.title('Scatter Plot of r_wspd vs r_apower')
plt.xlabel('wspd_70')
plt.ylabel('r_wspd')

# 保存为SVG格式，设置dpi为600
save_path = 'E:/HJHCloud/Seafile/startup/GoldWindPower/vision/pics/'
plt.savefig('fwind_rwind.svg', format='svg', dpi=600)

# 显示图形
plt.show()

