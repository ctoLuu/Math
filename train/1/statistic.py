import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_excel('./handledData2.xlsx')
print(df)
x = (df.loc[:,'CO(GT)'].to_numpy()).reshape((-1, 1))
y = df.loc[:,'C6H6(GT)'].to_numpy()
print(x)
print(y)
# 示例数据
# x = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
# y = np.array([2, 4, 6, 5, 10])
#
# # 生成多项式特征
poly = PolynomialFeatures(degree=1)  # 选择多项式的阶数
x_poly = poly.fit_transform(x)

# 多项式回归模型
model = LinearRegression()
model.fit(x_poly, y)

# 绘制原始数据点
# plt.scatter(x, y, color='red', label='原始数据')
plt.scatter(x, y, s=30, marker='o', facecolors='none', edgecolors='blue', label='数据点')
# 绘制拟合曲线
x_line = np.linspace(x.min(), x.max(), 100).reshape(100, 1)
x_line_poly = poly.transform(x_line)
y_pred = model.predict(x_line_poly)
plt.plot(x_line, y_pred, color='red', label='多项式回归')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()