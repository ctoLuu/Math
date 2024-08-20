import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 导入和处理数据
df = pd.read_excel('./origin_data.xlsx')
df = df.interpolate(method='linear')
data_CN = df['CN'].values[650000:]
data_SD = df['SD'].values[650000:]
print(data_CN)

# 定义ARIMA模型的阶数 (p, d, q)
order = (60, 1, 0)

# CN列的ARIMA模型拟合
model_CN = ARIMA(data_CN, order=order)
model_CN_fit = model_CN.fit()

# SD列的ARIMA模型拟合
model_SD = ARIMA(data_SD, order=order)
model_SD_fit = model_SD.fit()

# 预测未来一天负荷（5760个时间点，每15秒一个数据点）
forecast_CN = model_CN_fit.forecast(steps=5760)
forecast_SD = model_SD_fit.forecast(steps=5760)

# 绘制CN列的预测结果
plt.figure(figsize=(14, 5))
plt.plot(forecast_CN, label='Predicted CN Load')
plt.title('Predicted CN Load for the Next Day (Every 15s)')
plt.xlabel('Time (15s intervals)')
plt.ylabel('Load')
plt.legend()
plt.show()

# 绘制SD列的预测结果
plt.figure(figsize=(14, 5))
plt.plot(forecast_SD, label='Predicted SD Load')
plt.title('Predicted SD Load for the Next Day (Every 15s)')
plt.xlabel('Time (15s intervals)')
plt.ylabel('Load')
plt.legend()
plt.show()
