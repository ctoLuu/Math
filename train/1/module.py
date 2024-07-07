import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = "./out/result.csv"
df = pd.read_csv(file_path)
print(df.head(5))
df['销售日期'] = pd.to_datetime(df['销售日期'])
df.set_index('销售日期', inplace=True)
list_test = ['花叶类', '花菜类', '水生根茎类', '茄类', '辣椒类', '食用菌']

i = 4
print(i)
df = df[df['品类'] == list_test[i]]
# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['批发价格 (元/千克)'].values.reshape(-1, 1))
train_size = int(len(scaled_data) * 1)
train = scaled_data[0:train_size, :]

# 转换数据格式为符合 LSTM 输入要求
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])

	return np.array(dataX), np.array(dataY)


look_back = 30
trainX, trainY = create_dataset(train, look_back)
pre_X = train[-30:]
pre_X = [item for sublist in pre_X for item in sublist]
pre_X = np.array([[pre_X]])
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(6, input_shape=(1, look_back)))
model.add(Dense(7))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)

trainPredict = model.predict(trainX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

pre_Y = model.predict(pre_X)
pre_Y = scaler.inverse_transform(pre_Y)

# 计算 R 方、MSE、RMSE 和 MAE
mse = mean_squared_error(trainY[0], trainPredict[:, 0])
rmse = sqrt(mse)
mae = mean_absolute_error(trainY[0], trainPredict[:, 0])
r2 = r2_score(trainY[0], trainPredict[:, 0])
print(f'R2: {r2}, MSE: {mse}, RMSE: {rmse}, MAE: {mae}')


print(f" 预测未来七天{list_test[i]}的销量：{pre_Y[0, :]}")

# 创建一个 figure 和 axes 对象
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(trainY[0], label=f'{list_test[i]} 实际进价')
ax.plot(trainPredict[:, 0], label=f'{list_test[i]} 实际进价')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.set_title(f'{list_test[i]} LSTM 预测结果图')
ax.grid()
beautiful(ax)
fig.savefig(f"./rst2/{list_test[i]}Loss.png")
fig.show()

# 创建一个新的 figure 和 axes 对象
fig, ax = plt.subplots(figsize=(12, 6))
loss_history = history.history['loss']
ax.plot(loss_history, label='Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.set_title('Loss Over Training Epochs')
ax.grid()
beautiful(ax)
fig.savefig(f"./rst2/{list_test[i]}LSTM 预测结果图.png")
fig.show()