import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime, timedelta
import csv
from scipy.stats import kstest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scs

# 数据加载和预处理函数
def load_and_preprocess_data(filepath, center_id):
    # 读取数据并转换日期时间格式
    data = pd.read_csv(filepath, encoding='GB2312')
    data['日期'] = pd.to_datetime(data['日期'])
    data['小时'] = data['小时'].astype(int)  # 确保小时是整数类型
    # 补充缺失值
    data_center = data[data['分拣中心'] == center_id]
    # 将日期和小时合并为一个新的日期时间索引
    data_center['DateTime'] = pd.to_datetime(data_center['日期'].astype(str) + '-' + data_center['小时'].astype(str).str.zfill(2))
    # 将DateTime设置为新的索引
    data_center.set_index('DateTime', inplace=True)
    # 确保数据按照日期时间索引排序
    data_center = data_center.sort_index()
    # 重新采样以填补缺失的小时数据，并进行线性插值
    resampled_data = data_center.resample('H').interpolate(method='linear')
    # 缩放货量数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    resampled_data['scaled_货量'] = scaler.fit_transform(resampled_data[['货量']])
    # 重置索引，以便后续处理
    resampled_data.reset_index(inplace=True)
    return resampled_data[['分拣中心', '日期', '小时', '货量', 'scaled_货量']], scaler

# 创建数据集
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset.iloc[i:(i + look_back)][['小时', 'scaled_货量']].values
        X.append(a)
        Y.append(dataset.iloc[i + look_back]['scaled_货量'])
    return np.array(X), np.array(Y)


# 构建并训练模型
def build_and_train_model(data_center, look_back=24):
    X, Y = create_dataset(data_center, look_back)
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 2), dropout=0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=60, batch_size=1, verbose=2)
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
data_path = '附件2.csv'
data = pd.read_csv(data_path, encoding='GB2312')

# 保存预测结果到CSV文件
results_csv_path = '结果表2.csv'
with open(results_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['分拣中心', '日期', '小时', '货量'])
    for center_id in data['分拣中心'].unique():
        print(f"Processing {center_id}...")
        data_center, scaler = load_and_preprocess_data(data_path, center_id)
        model = build_and_train_model(data_center, look_back=24)
        last_24_hours = data_center.tail(24)[['小时', 'scaled_货量']].values.reshape(1, 24, 2)
        predictions = predict_next_30_days(model, last_24_hours, scaler)
        # 写入CSV
        start_date = data_center['日期'].max() + timedelta(days=1)
        for i in range(30 * 24):
            predict_date = start_date + timedelta(hours=i)
            writer.writerow([center_id, predict_date.strftime('%Y-%m-%d'), predict_date.hour, predictions[i][0]])
print("预测结果已保存到", results_csv_path)

data['日期'] = pd.to_datetime(data['日期'])
data2_path = "结果表2.csv"
data2 = pd.read_csv(data2_path, encoding='GB2312')
data2['日期'] = data2['日期'].astype(str)
data2['小时'] = data2['小时'].astype(int)
data2['日期'] = pd.to_datetime(data2['日期'], errors='coerce')
# 合并原始及预测数据
df = pd.concat([data, data2], ignore_index=True)
df.sort_values(['日期', '小时'], inplace=True)
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values(by=['日期', '小时'])

df['小时'] = df['小时'].astype(str).str.pad(side='left', fillchar='0', width=2)

# 合并日期和小时
df['日期时间'] = df['日期'].astype(str) + ' ' + df['小时']
df['日期时间'] = pd.to_datetime(df['日期时间'])
df['日期时间'] = pd.to_datetime(df['日期时间'], errors='coerce')
df.drop(['日期', '小时'], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
df.rename(columns={'日期时间': '日期小时'}, inplace=True)


CSs = df['分拣中心'].unique()[0:5]
tem = pd.DataFrame()
for i in CSs:
    # 获取当前分拣中心的数据
    current_center_data = df[df['分拣中心'] == i]
    # 确保'日期时间'列是日期时间类型
    current_center_data['日期小时'] = pd.to_datetime(current_center_data['日期小时'])
    # 筛选出12月之前的数据
    pre_december_data = current_center_data[current_center_data['日期小时'] < datetime(2023, 12, 1)].copy()
    pre_december_data['颜色'] = 'red'  # 为12月前的数据指定颜色
    # 筛选出12月之后的数据
    post_december_data = current_center_data[current_center_data['日期小时'] >= datetime(2023, 12, 1)].copy()
    post_december_data['颜色'] = 'blue'  # 为12月后的数据指定颜色
    # 将两部分数据合并，并添加到tem中
    tem = pd.concat([tem, pre_december_data, post_december_data], ignore_index=True)
# 创建 FacetGrid 对象
g = sns.FacetGrid(data=tem, row="分拣中心", hue="颜色", height=2.5, aspect=6, sharex=False, sharey=False)
# 绘制线图
g = g.map(sns.lineplot, "日期小时", "货量", palette="Set1", lw=2)
# 设置x轴标签
plt.xlabel('时间')
# 显示图形
plt.show()