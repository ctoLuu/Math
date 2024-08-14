from datetime import datetime, timedelta
from meteostat import Point, Hourly
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

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

# 将数据保存以便检查
data.to_csv('hangzhou_xiasha_hourly_weather.csv')

# 2. 数据预处理
# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 创建训练集
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 24  # 使用前24小时的数据来预测下一个小时
X, y = create_dataset(scaled_data, time_step)

# 将数据 reshape 成 LSTM 需要的格式 [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. 构建并训练 LSTM 模型
# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, batch_size=64, epochs=500, validation_data=(X_test, y_test))

# 4. 预测2024年8月12日到9月8日的温度
# 首先获取从8月12日开始的数据并进行预测
n_hours_to_predict = (datetime(2024, 9, 8) - datetime(2024, 8, 12)).days * 24
input_data = scaled_data[-time_step:]  # 使用最后24小时的数据

# 准备数据进行预测
temp_input = list(input_data)
temp_input = temp_input[0:].copy()

predictions_future = []

for i in range(n_hours_to_predict):
    x_input = np.array(temp_input[-time_step:])
    x_input = x_input.reshape(1, time_step, 1)
    yhat = model.predict(x_input, verbose=0)
    predictions_future.append(yhat[0][0])
    temp_input.append(yhat[0])

# 反归一化预测数据
predictions_future_aug_to_sept = scaler.inverse_transform(np.array(predictions_future).reshape(-1, 1))

# 生成预测时间段的日期索引
future_dates_aug_to_sept = pd.date_range(start='2024-08-12 00:00:00', periods=n_hours_to_predict, freq='H')

# 绘制从8月12日到9月8日的预测结果
plt.figure(figsize=(16, 8))
plt.plot(future_dates_aug_to_sept, predictions_future_aug_to_sept, label='Predicted Temperature (Aug 12 - Sept 8)')
plt.title('Hourly Temperature Prediction for Aug 12 - Sept 8, 2024')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# 5. 预测9月8日到9月10日的温度
# 使用前面的预测值来进一步预测9月8日到9月10日的温度
n_hours_to_predict_further = 72  # 预测两天的数据（48小时）
input_data = predictions_future_aug_to_sept[-time_step:]  # 使用预测的最后24小时的数据

# 准备数据进行预测
temp_input = list(input_data)
temp_input = temp_input[0:].copy()

predictions_future_sept = []

for i in range(n_hours_to_predict_further):
    x_input = np.array(temp_input[-time_step:])
    x_input = x_input.reshape(1, time_step, 1)
    yhat = model.predict(x_input, verbose=0)
    predictions_future_sept.append(yhat[0][0])
    temp_input.append(yhat[0])

# 反归一化预测数据
predictions_future_sept = scaler.inverse_transform(np.array(predictions_future_sept).reshape(-1, 1))

# 生成预测时间段的日期索引
future_dates_sept = pd.date_range(start='2024-09-08 00:00:00', periods=n_hours_to_predict_further, freq='H')

# 绘制从9月8日到9月10日的预测结果
plt.figure(figsize=(16, 8))
plt.plot(future_dates_sept, predictions_future_sept, label='Predicted Temperature (Sept 8-10)')
plt.title('Hourly Temperature Prediction for Sept 8-10, 2024')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
