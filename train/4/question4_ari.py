from datetime import datetime
from meteostat import Point, Hourly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 1. 获取杭州下沙的历史每小时温度数据
# 定义地点（杭州下沙的经纬度）
hangzhou_xiasha = Point(30.3082, 120.3681)

# 定义时间范围
start = datetime(2019, 1, 1)
end = datetime(2024, 8, 12)

# 获取每小时的天气数据
data = Hourly(hangzhou_xiasha, start, end)
data = data.fetch()

# 使用温度列，并处理缺失值
data = data[['temp']]
data.fillna(method='ffill', inplace=True)

# 2. 划分训练集和测试集
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# 3. 构建并训练 ARIMA 模型
# ARIMA(p,d,q) 模型的阶数选择可以通过 AIC/BIC 或通过经验进行调整
# 这里假设使用 (p=5, d=1, q=0) 作为初始值
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# 4. 使用模型进行预测
# 对测试集进行预测
predictions = model_fit.forecast(steps=len(test))
predictions = pd.Series(predictions, index=test.index)

# 计算预测误差
error = mean_squared_error(test, predictions)
print(f'Test Mean Squared Error: {error}')

# 绘制预测结果和真实值
plt.figure(figsize=(16, 8))
plt.plot(test.index, test, label='True Temperature')
plt.plot(predictions.index, predictions, label='Predicted Temperature')
plt.title('Temperature Prediction on Test Data')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# 5. 预测2024年9月8日到9月10日的温度
# 使用整个数据集重新训练模型并预测未来
full_model = ARIMA(data, order=(5, 1, 0))
full_model_fit = full_model.fit()

# 预测未来48小时的温度
n_hours = 72
future_predictions = full_model_fit.forecast(steps=n_hours)

# 生成预测时间段的日期索引
future_dates = pd.date_range(start='2024-09-08 00:00:00', periods=n_hours, freq='H')

# 绘制未来预测结果
plt.figure(figsize=(16, 8))
plt.plot(future_dates, future_predictions, label='Predicted Temperature (Sept 8-10)')
plt.title('Hourly Temperature Prediction for Sept 8-10, 2024')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
