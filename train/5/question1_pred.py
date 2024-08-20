import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


df = pd.read_excel('./origin_data.xlsx')
df = df.interpolate(method='linear')
print(df)

data_CN = df['CN'].values
data_SD = df['SD'].values

# 归一化数据
scaler_CN = MinMaxScaler(feature_range=(0, 1))
data_CN = scaler_CN.fit_transform(data_CN.reshape(-1, 1))

scaler_SD = MinMaxScaler(feature_range=(0, 1))
data_SD = scaler_SD.fit_transform(data_SD.reshape(-1, 1))

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - 2*time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps:i+time_steps*2])
    print(np.array(X))
    print(np.array(y))
    return np.array(X), np.array(y)

time_steps = 60  # 15分钟的窗口，每15秒一个数据点
X_CN, y_CN = create_sequences(data_CN, time_steps)
X_SD, y_SD = create_sequences(data_SD, time_steps)

X_CN_train, X_CN_test, y_CN_train, y_CN_test = train_test_split(X_CN, y_CN, test_size=0.2, shuffle=False)
X_SD_train, X_SD_test, y_SD_train, y_SD_test = train_test_split(X_SD, y_SD, test_size=0.2, shuffle=False)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 对CN建立模型
model_CN = build_lstm_model((X_CN_train.shape[1], 1))

# 对SD建立模型
model_SD = build_lstm_model((X_SD_train.shape[1], 1))

# 训练CN模型
model_CN.fit(X_CN_train, y_CN_train, batch_size=64, epochs=5, validation_data=(X_CN_test, y_CN_test))

# 训练SD模型
model_SD.fit(X_SD_train, y_SD_train, batch_size=64, epochs=5, validation_data=(X_SD_test, y_SD_test))

# 预测CN的未来一天负荷
predicted_CN = []
last_sequence_CN = X_CN_test[-1]

for i in range(24 * 60 * 60 // 15):  # 一天的15秒数据
    next_pred_CN = model_CN.predict(last_sequence_CN.reshape(1, -1, 1))
    predicted_CN.append(next_pred_CN[0, 0])
    last_sequence_CN = np.roll(last_sequence_CN, -1)
    last_sequence_CN[-1] = next_pred_CN

predicted_CN = scaler_CN.inverse_transform(np.array(predicted_CN).reshape(-1, 1))

# 预测SD的未来一天负荷
predicted_SD = []
last_sequence_SD = X_SD_test[-1]

for i in range(24 * 60 * 60 // 15):
    next_pred_SD = model_SD.predict(last_sequence_SD.reshape(1, -1, 1))
    predicted_SD.append(next_pred_SD[0, 0])
    last_sequence_SD = np.roll(last_sequence_SD, -1)
    last_sequence_SD[-1] = next_pred_SD

predicted_SD = scaler_SD.inverse_transform(np.array(predicted_SD).reshape(-1, 1))

import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))
plt.plot(predicted_CN, label='Predicted CN Load')
plt.title('Predicted CN Load for the Next Day (Every 15s)')
plt.xlabel('Time (15s intervals)')
plt.ylabel('Load')
plt.legend()
plt.show()

plt.figure(figsize=(14,5))
plt.plot(predicted_SD, label='Predicted SD Load')
plt.title('Predicted SD Load for the Next Day (Every 15s)')
plt.xlabel('Time (15s intervals)')
plt.ylabel('Load')
plt.legend()
plt.show()

predicted_data = pd.DataFrame({
    'Predicted_CN': predicted_CN.flatten(),
    'Predicted_SD': predicted_SD.flatten()
})

# 保存到 Excel 文件
predicted_data.to_excel('predicted_results.xlsx', index=False)

print("预测结果已保存到 predicted_results.xlsx 文件中")