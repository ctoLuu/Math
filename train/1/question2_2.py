import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv
from scipy.stats import kstest
import seaborn as sns
from scipy import stats as scs
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 创建数据集
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset.iloc[i:(i + look_back)][['Time', 'NMHC(GT)']].values
        X.append(a)
        Y.append(dataset.iloc[i + look_back]['NMHC(GT)'])
    print(len(X))
    print(len(Y))
    return np.array(X), np.array(Y)


# 构建并训练模型
def build_and_train_model(data_center, look_back=168):
    X, Y = create_dataset(data_center, look_back)
    model = Sequential()
    model.add(LSTM(40, input_shape=(look_back, 2), dropout=0.2))
    model.add(Dense(10))
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=400, batch_size=1, verbose=2)
    return model


# 进行未来30天每小时预测
def predict_next_30_days(model, last_24_hours, scaler, look_back=24):
    predictions = []
    current_batch = last_24_hours
    for i in range(30 * 24):  # 30天每天24小时
        current_pred = model.predict(current_batch)[0][0]  # 获取模型预测的单个值
        predictions.append(current_pred)
        next_hour = (current_batch[0, -1, 0] + 1) % 24  # 更新小时
        # 确保新的行是正确类型
        new_row = np.array([[next_hour, current_pred]], dtype=np.float32).reshape(1, 2)
        # 更新批次数据
        current_batch = np.append(current_batch[:, 1:, :], new_row.reshape(1, 1, 2), axis=1)
    # 反归一化预测结果
    inversed_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    # 将预测<0的数据置为0
    return np.maximum(inversed_predictions, 0)

# 主逻辑
results_csv_path = '结果表.csv'
with open(results_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Time', 'NMHC(GT)'])
    df = pd.read_excel('./handledData4.xlsx')
    print(df)
    df = df[0:9355]
    df['Time'] = df['Time'].str.slice(start=0, stop=2).astype(int)
    print(df)
    df['日期时间'] = df['Date'].astype(str) + '-' + df['Time'].astype(str)
    df['日期时间'] = pd.to_datetime(df['日期时间'], errors='coerce')
    df.rename(columns={'日期时间': '日期小时'}, inplace=True)
    df.set_index('日期小时', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['NMHC(GT)'] = scaler.fit_transform(df[['NMHC(GT)']])
    print(f"Processing ...")
    data_center = df[['Date', 'Time', 'NMHC(GT)']]
    print(data_center)
    model = build_and_train_model(data_center, look_back=168)
    last_24_hours = data_center.tail(24)[['Time', 'NMHC(GT)']].values.reshape(1, 24, 2)
    predictions = predict_next_30_days(model, last_24_hours, scaler)
    start_date = data_center['Date'].max() + timedelta(days=1)
    for i in range(30 * 24):
        predict_date = start_date + timedelta(hours=i)
        writer.writerow([predict_date.strftime('%Y-%m-%d'), predict_date.hour, predictions[i][0]])
print("预测结果已保存到", results_csv_path)

df = pd.read_excel('./handledData4.xlsx')
df = df[0:9355]
df['Time'] = df['Time'].str.slice(start=0, stop=2).astype(int)
df['Date'] = pd.to_datetime(df['Date'])
data2_path = "结果表.csv"
data2 = pd.read_csv(data2_path, encoding='GB2312')
data2['Date'] = data2['Date'].astype(str)
data2['Time'] = data2['Time'].astype(int)
data2['Date'] = pd.to_datetime(data2['Date'], errors='coerce')
# 合并原始及预测数据
df = pd.concat([df, data2], ignore_index=True)
df.sort_values(['Date', 'Time'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date', 'Time'])

df['Time'] = df['Time'].astype(str).str.pad(side='left', fillchar='0', width=2)

# 合并日期和小时
df['日期时间'] = df['Date'].astype(str) + ' ' + df['Time']
df['日期时间'] = pd.to_datetime(df['日期时间'])
df['日期时间'] = pd.to_datetime(df['日期时间'], errors='coerce')
df.drop(['Date', 'Time'], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
df.rename(columns={'日期时间': '日期小时'}, inplace=True)
plt.figure(figsize=(30, 20))
tem = pd.DataFrame()
# 确保'日期时间'列是日期时间类型
df['日期小时'] = pd.to_datetime(df['日期小时'])
# 筛选出12月之前的数据
pre_december_data = df[df['日期小时'] < datetime(2005, 4, 4)].copy()
pre_december_data['颜色'] = 'red'  # 为12月前的数据指定颜色
# 筛选出12月之后的数据
post_december_data = df[df['日期小时'] >= datetime(2005, 4, 4)].copy()
post_december_data['颜色'] = 'blue'  # 为12月后的数据指定颜色
# 将两部分数据合并，并添加到tem中
tem = pd.concat([tem, pre_december_data, post_december_data], ignore_index=True)
# 创建 FacetGrid 对象
g = sns.FacetGrid(data=tem, hue="颜色", height=4, aspect=6, sharex=False, sharey=False)
# 绘制线图
g = g.map(sns.lineplot, "日期小时", "NMHC(GT)", palette="Set1", lw=1)
# sns.lineplot(data=tem, x='日期小时', y='NMHC(GT)', palette="Set1", lw=2.5)
# 设置x轴标签
plt.xlabel('Time')
# 显示图形
plt.show()
