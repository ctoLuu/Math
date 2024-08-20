from datetime import datetime
from meteostat import Point, Hourly
import pandas as pd
import matplotlib.pyplot as plt

# 定义地点（杭州下沙的经纬度）
hangzhou_xiasha = Point(30.3082, 120.3681)

# 定义时间范围
start = datetime(2019, 1, 1)
end = datetime(2024, 8, 12)

# 获取每小时的天气数据
data = Hourly(hangzhou_xiasha, start, end)
data = data.fetch()

# 打印前几行数据
print(data.head())

# 将数据保存到 CSV 文件
data.to_csv('hangzhou_xiasha_hourly_weather.csv')
print("数据已保存至 'hangzhou_xiasha_hourly_weather.csv'")

# 计算每天的平均温度
daily_avg = data.resample('D').mean()

# 绘制温度变化图表
plt.figure(figsize=(14, 7))
plt.plot(daily_avg.index, daily_avg['temp'], label='Average Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
