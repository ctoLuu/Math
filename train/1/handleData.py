import pandas as pd
from scipy.stats import kstest
import numpy as np

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
    count = 0
    if ks_res == 1:
        # 定义3σ法则识别异常值
        lower_bound = u - 3 * std
        upper_bound = u + 3 * std
        # 将异常值替换为np.nan
        count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        print('异常值个数')
        print(count)
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
        return df
    elif ks_res == 0:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 将异常值替换为np.nan
        count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        print('异常值个数')
        print(count)
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
        return df
    else:
        # 如果ks_res既不是0也不是1，直接返回原始数据框
        return df


def handle_consecutive_minus(series, column_list):
    for column in column_list:
        count = 0  # 用来记录连续-200的个数
        flag = None
        print(len(series[column]))
        for i in range(len(series[column])):
            if series.loc[i, column] == -200:
                count += 1
                if count == 1:
                    flag = i
            else:
                if count >= 3:
                    series.loc[flag:i-1, column] = np.nan
                elif count != 0:
                    series.loc[flag:i-1, column] = np.nan
                count = 0
                flag = None
        df.reset_index(drop=True, inplace=True)
        print(series)

    return series


def handle_minus(series, column_list):
    for column in column_list:
        count = 0  # 用来记录连续-200的个数
        flag = None
        print(len(series[column]))
        for i in range(len(series[column])):
            if series.loc[i, column] == -200:
                count += 1
                if count == 1:
                    flag = i
            else:
                # if count >= 3:
                    # series.drop(index=range(flag, i), inplace=True)  # 删除索引范围内的行
                if count != 0 and count < 3:
                    series.loc[flag:i-1, column] = np.nan
                count = 0
                flag = None
        # 对NaN值进行线性插值
        series[column] = series[column].interpolate(method='linear')
        df.reset_index(drop=True, inplace=True)
        print(series)


    return series


def add_data(series):
    count = 0
    flag = 0
    for column in list(series)[2:]:

        for i in range(len(series[column])):
            if pd.isna(series.loc[i, column]):
                count += 1
                if count == 1:
                    flag = i
            else:
                if count != 0 and count <= 6:
                    series.loc[flag-1:i, column] = series.loc[flag-1:i, column].interpolate(method='linear')
                count = 0
                flag = 0
        # 对NaN值进行线性插值
        print(series)
    return series


# 示例使用
df = pd.read_excel('./AirQualityUCI.xlsx')
print(df)
df = handle_minus(df, list(df)[2:])
df = handle_consecutive_minus(df, list(df)[2:])
for column in list(df)[2:]:
    ks_res = KsNormDetect(df, column)
    df = OutlierDetection(df, column, ks_res)
print(df)
df = add_data(df)
df.to_excel('handledData3.xlsx', index=False)