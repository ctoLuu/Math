import pandas as pd
from scipy.stats import kstest
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # 导入dates模块
plt.rcParams['font.sans-serif'] = ['SimHei']
def KsNormDetect(df, column):
    # 计算均值
    u = df[column].mean()
    # 计算标准差
    std = df[column].std()
    # 计算P值
    res = kstest(df[column], 'norm', (u, std))[1]
    # 判断p值是否服从正态分布，p<=0.05 则服从正态分布，否则不服从。
    if res <= 0.05:
        print('该列数据服从正态分布------------')
        print('均值为：%.3f，标准差为：%.3f' % (u, std))
        print('------------------------------')
        return 1
    else:
        return 0


def OutlierDetection(df, column, ks_res):
    # 计算均值
    u = df[column].mean()
    # 计算标准差
    std = df[column].std()

    if ks_res == 1:
        # 定义3σ法则识别异常值
        lower_bound = u - 3 * std
        upper_bound = u + 3 * std
        # 将异常值替换为np.nan
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
        return df
    elif ks_res == 0:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 将异常值替换为np.nan
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
        return df
    else:
        # 如果ks_res既不是0也不是1，直接返回原始数据框
        return df


def handle_consecutive_minus_200(series):
    count = 0  # 用来记录连续-200的个数
    flag = 0
    mount = 0
    print(len(series))
    for i in range(len(series)):
        mount += 1
        print(mount)
        print(i)
        if series.loc[i, "CO(GT)"] == -200:
            count += 1
            if count == 1:
                flag = i
        else:
            if count >= 3:
                series.drop(index=range(flag, i), inplace=True)  # 删除索引范围内的行
            elif count != 0:
                series.loc[flag:i-1, "CO(GT)"] = np.nan
            count = 0  # 重置计数器
    print("\n\n\n\n")
    count = 0
    for i in range(len(series)):
        mount += 1
        print(mount)
        if i not in series.index:
            continue
        if series.loc[i, "C6H6(GT)"] == -200:
            count += 1
            if count == 1:
                flag = i
        else:
            if count >= 3:
                series.drop(index=range(flag, i), inplace=True)  # 删除索引范围内的行
            elif count != 0:
                series.loc[flag:i-1, "C6H6(GT)"] = np.nan
            count = 0  # 重置计数器
    # 对NaN值进行线性插值
    series['CO(GT)'] = series['CO(GT)'].interpolate(method='linear')
    series['C6H6(GT)'] = series['C6H6(GT)'].interpolate(method='linear')
    return series


df = pd.read_excel('./AirQualityUCI.xlsx')
print(df)
df = handle_consecutive_minus_200(df)
print(df)
for column in list(df)[2:]:
    ks_res = KsNormDetect(df, column)
    df = OutlierDetection(df, column, ks_res)
df['CO(GT)'] = df['CO(GT)'].interpolate(method='linear')
df['C6H6(GT)'] = df['C6H6(GT)'].interpolate(method='linear')
X = df[['CO(GT)']]  # 自变量，用列名替换'feature'
y = df[['C6H6(GT)']]     # 因变量，用列名替换'target'
X = X.values.reshape(-1, 1)

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 获取模型参数，即斜率和截距
slope = model.coef_[0]
intercept = model.intercept_

# 打印模型参数
print("斜率:", slope)
print("截距:", intercept)

# 绘制数据点
plt.scatter(X, y, s=30, marker='o', facecolors='none', edgecolors='blue', label='数据点')
# 使用模型参数绘制拟合线
X_line = np.array([np.min(X), np.max(X)]).reshape(-1, 1)  # 生成拟合线需要的X值范围
y_line = intercept + slope * X_line  # 计算对应的y值
plt.plot(X_line, y_line, color='red', label='拟合线')
# 添加图例
plt.legend()
# 设置图表标题和轴标签
plt.title('线性回归分析')
plt.xlabel('CO(GT)')
plt.ylabel('C6H6(GT)')
# 显示图表
plt.show()
df.to_excel('handledData2.xlsx', index=False)
