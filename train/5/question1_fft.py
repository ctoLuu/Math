import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. 数据加载和预处理
df = pd.read_excel('./origin_data_two.xlsx')
df = df.interpolate(method='linear')
print(df.head())

data_CN = df['CN'].values[270720:]  # 599040 639360 633600 *627840 *668160 *673920
data_SD = df['SD'].values[270720:]  # 599040 639360 633600 *627840 *668160 *673920

# 460800 *455040
# *270720
# 归一化数据
scaler_CN = MinMaxScaler(feature_range=(0, 1))
data_CN = scaler_CN.fit_transform(data_CN.reshape(-1, 1)).flatten()

scaler_SD = MinMaxScaler(feature_range=(0, 1))
data_SD = scaler_SD.fit_transform(data_SD.reshape(-1, 1)).flatten()

# 2. 快速傅里叶变换（FFT）分析
def apply_fft(data, sampling_rate=1):
    fft_result = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), d=sampling_rate)

    # 只保留正频率分量
    positive_freqs = fft_freq[np.where(fft_freq >= 0)]
    positive_fft_result = np.abs(fft_result[np.where(fft_freq >= 0)])

    return positive_freqs, positive_fft_result, fft_result

# 设置采样率（每15秒一个数据点）
sampling_rate = 1 / 15

# 对 CN 数据进行 FFT 分析
freqs_CN, fft_values_CN, fft_result_CN = apply_fft(data_CN, sampling_rate)

# 对 SD 数据进行 FFT 分析
freqs_SD, fft_values_SD, fft_result_SD = apply_fft(data_SD, sampling_rate)

# 3. 信号重构
def reconstruct_signal(fft_result, num_components=5760):
    indices = np.argsort(np.abs(fft_result))[-num_components:]
    filtered_fft = np.zeros_like(fft_result)
    filtered_fft[indices] = fft_result[indices]
    return np.fft.ifft(filtered_fft).real

# 重构 CN 和 SD 信号
reconstructed_cn_signal = reconstruct_signal(fft_result_CN)
reconstructed_sd_signal = reconstruct_signal(fft_result_SD)

# 4. 预测未来5760个数据点
# 获取重构信号的长度和周期
signal_length = len(reconstructed_cn_signal)
predicted_cn_signal = np.tile(reconstructed_cn_signal, 5760 // signal_length + 1)[:5760]
predicted_sd_signal = np.tile(reconstructed_sd_signal, 5760 // signal_length + 1)[:5760]

# 反归一化数据
predicted_cn_signal = scaler_CN.inverse_transform(predicted_cn_signal.reshape(-1, 1)).flatten()
predicted_sd_signal = scaler_SD.inverse_transform(predicted_sd_signal.reshape(-1, 1)).flatten()

# 5. 保存预测数据到 Excel 文件
predicted_data = pd.DataFrame({
    'Predicted_CN': predicted_cn_signal,
    'Predicted_SD': predicted_sd_signal
})

predicted_data.to_excel('predicted_results_fft.xlsx', index=False)
print("预测结果已保存到 predicted_results_fft.xlsx 文件中")

# 6. 可视化预测结果
plt.figure(figsize=(14, 5))
plt.plot(predicted_cn_signal, label='Predicted CN Signal')
plt.xlabel('Time (15s intervals)')
plt.ylabel('Load')
plt.legend()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(predicted_sd_signal, label='Predicted SD Signal')
plt.xlabel('Time (15s intervals)')
plt.ylabel('Load')
plt.legend()
plt.show()
