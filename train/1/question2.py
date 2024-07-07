import pandas as pd
from scipy.stats import kstest
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # 导入dates模块


plt.rcParams['font.sans-serif'] = ['SimHei']
df = pd.read_excel('./handledData3.xlsx')
df = df[0:184]
X = df[['C6H6(GT)']]  # 自变量，用列名替换'feature'
y = df[['NMHC(GT)']]     # 因变量，用列名替换'target'

# 将X转换为二维数组，因为scikit-learn需要这样的格式
X = X.values.reshape(-1, 1)

# 创建线性回归模型实例
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 获取模型参数，即斜率和截距
slope = model.coef_[0]
intercept = model.intercept_

# 打印模型参数
print("斜率:", slope)
print("截距:", intercept)
#
# # 绘制数据点
# plt.scatter(X, y, s=30, marker='o', facecolors='none', edgecolors='blue', label='数据点')
#
#
# # 使用模型参数绘制拟合线
X_line = np.array([np.min(X), np.max(X)]).reshape(-1, 1)  # 生成拟合线需要的X值范围
y_line = intercept + slope * X_line  # 计算对应的y值
plt.plot(X_line, y_line, color='red', label='拟合线', linewidth=3)
#
# # 添加图例
# plt.legend()
#
# # 设置图表标题和轴标签
# plt.title('线性回归分析')
# plt.xlabel('C6H6(GT)')
# plt.ylabel('NMHC(GT)')
#
# # 显示图表
# plt.show()

df = pd.read_excel('./handledData3.xlsx')
for i in range(185, 9357):
    if pd.isna(df['NMHC(GT)'][i]):
        new = df['C6H6(GT)'][i] * slope + intercept;
        if new < 0: new = 0
        df['NMHC(GT)'][i] = new

df.to_excel('handledData4.xlsx', index=False)
# ... 您之前的代码 ...

# 绘制数据点
plt.scatter(X, y, s=30, marker='o', facecolors='none', edgecolors='blue', label='原始数据点')

# 假设您有另一组数据，我们将其命名为X_new和y_new
# 您可以这样添加新的散点
X_new = df[['C6H6(GT)']][185:9357].values.reshape(-1, 1)  # 假设这是新数据的X值
y_new = df['NMHC(GT)'][185:9357]  # 这是新数据的y值

# 过滤掉NaN值
mask = ~np.isnan(y_new)
X_new = X_new[mask]
y_new = y_new[mask]

# 计算新数据的预测值
y_pred = intercept + slope * X_new

# 绘制新数据的散点，使用不同的颜色
plt.scatter(X_new, y_new, s=30, marker='o', facecolors='none', edgecolors='green', label='新数据点')

# 绘制新数据的预测线
X_line_new = np.array([np.min(X_new), np.max(X_new)]).reshape(-1, 1)
y_line_new = intercept + slope * X_line_new

# 添加图例
plt.legend()

# 设置图表标题和轴标签
plt.title('线性回归分析')
plt.xlabel('C6H6(GT)')
plt.ylabel('NMHC(GT)')

# 显示图表
plt.show()



