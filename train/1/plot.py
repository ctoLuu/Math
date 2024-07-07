import pandas as pd
from scipy.stats import kstest
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # 导入dates模块

df = pd.read_excel('./handledData3.xlsx')
df['日期时间'] = df['Date'].astype(str) + '-' + df['Time'].astype(str)
df['日期时间'] = pd.to_datetime(df['日期时间'], errors='coerce')
df.drop(['Date', 'Time'], axis=1, inplace=True)
df.rename(columns={'日期时间': '日期小时'}, inplace=True)
print(df)
df.set_index('日期小时', inplace=True)
plt.rcParams.update({'font.size': 60})
# 绘制数据随时间变化的图
plt.figure(figsize=(50, 25))  # 可以设置图表的大小
plt.plot(df.index[0:168], df['NO2(GT)'][0:168], marker='o')  # 使用线和点来绘制

# 设置图表标题和轴标签
plt.xlabel('Date Time', fontsize=70)
plt.ylabel('NO2(GT)',fontsize=70)

# 格式化x轴的时间显示
plt.gcf().autofmt_xdate()  # 自动旋转日期标记
plt.gca().xaxis.set_tick_params(labelsize=30)
# 显示图表
plt.show()