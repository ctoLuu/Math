import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = '预测结果_单柜_2024-03-18.xlsx'  # 将此路径替换为你的文件路径
df = pd.read_excel(file_path)

# 假设真实值和预测值列名为 '真实值' 和 '预测值'
real_values = df['FH']
predicted_values = df['FH_预测']

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(real_values, label='real', color='blue', linewidth=2)
plt.plot(predicted_values, label='predict', color='red', linestyle='--', linewidth=2)
plt.xlabel('time')
plt.ylabel('data')
plt.legend()
plt.grid(True)
plt.show()
