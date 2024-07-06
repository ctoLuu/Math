import csv
import numpy as np
import pandas as pd
from scipy.stats import kstest
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy import stats as scs
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

sns.set(font="simhei",style="whitegrid",font_scale=1.6)
import warnings
warnings.filterwarnings("ignore")
def KsNormDetect(df):
    # 计算均值
    u = df['货量'].mean()
    # 计算标准差
    std = df['货量'].std()
    # 计算P值
    res = kstest(df['货量'], 'norm', (u, std))[1]
    # 判断p值是否服从正态分布，p<=0.05 则服从正态分布，否则不服从。
    if res <= 0.05:
        print('该列数据服从正态分布------------')
        print('均值为：%.3f，标准差为：%.3f' % (u, std))
        print('------------------------------')
        return 1
    else:
        return 0

def OutlierDetection(df, ks_res):
    # 计算均值
    u = df['货量'].mean()
    # 计算标准差
    std = df['货量'].std()
    if ks_res == 1:
        # 定义3σ法则识别异常值
        # 识别异常值
        error = df[np.abs(df['货量'] - u) > 3 * std]
        # 剔除异常值，保留正常的数据
        data_c = df[np.abs(df['货量'] - u) <= 3 * std]
        return data_c
    if ks_res == 0:
        Q1 = df['货量'].quantile(0.25)
        Q3 = df['货量'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (df['货量'] >= lower_bound) & (df['货量'] <= upper_bound)
        cleaned_df = df[mask]
    return cleaned_df


def perform_adf_test(timeseries):
    # 执行ADF检验
    result = adfuller(timeseries, autolag='AIC')
    return result

# 读取数据
csv_file_path = 'file1.csv'
df1 = pd.read_csv('file1.csv', encoding='gbk')
df2 = pd.read_csv('附件2.csv', encoding='gbk')
df3 = pd.read_csv('附件3.csv', encoding='gbk')
df4 = pd.read_csv('附件4.csv', encoding='gbk')

df1['日期'] = pd.to_datetime(df1['日期'])
df2['日期'] = pd.to_datetime(df2['日期'])
df1 = df1.sort_values(by='日期')
df2 = df2.sort_values(by=['日期', '小时'])
# 数据预处理
CSs = df1['分拣中心'].unique()
processed_dfs = []
for CS in CSs:
    # 筛选出属于当前分拣中心的所有数据
    grouped_df = df1[df1['分拣中心'] == CS]
    # 将筛选后的DataFrame存储到字典中，键为分拣中心的名称
    ks_result = KsNormDetect(grouped_df)
    # 如果数据服从正态分布，则进行异常值检测
    if ks_result == 1:
        # 剔除异常值
        cleaned_df = OutlierDetection(grouped_df, ks_result)
    else:
        cleaned_df = OutlierDetection(grouped_df, ks_result)
    processed_dfs.append(cleaned_df)
final_df = pd.concat(processed_dfs, ignore_index=True)
# 绘制原始数据折现图
CSs = final_df['分拣中心'].unique()[:]
tem = pd.DataFrame()
for i in CSs:
    tem = pd.concat([tem, final_df[final_df.分拣中心==i]])
g = sns.FacetGrid(data=tem, row="分拣中心", hue="分拣中心", height=2.5, aspect=6, sharex=False, sharey=False)
g = g.map(sns.lineplot, "日期", "货量", palette="Set1", lw=2)
plt.xlabel('时间')
plt.show()
# 创建ARIMA模型的参数设置
ar_params = (32, 0, 16)
future_df = pd.DataFrame()
all_dates = pd.date_range(start=final_df['日期'].min(), end=final_df['日期'].max(), freq='D')
for CS in final_df['分拣中心'].unique():
    # 获取当前分拣中心的货量数据
    current_center_data = final_df[final_df['分拣中心'] == CS]['货量'].values
    last_date = final_df[final_df['分拣中心'] == CS].iloc[-1]['日期']
    # 执行ADF检验
    adf_result = perform_adf_test(current_center_data)
    model = ARIMA(current_center_data, order=ar_params)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)  # 提取预测值
    future_data = forecast.tolist()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_data), freq='D')
    # 将预测数据与新的时间序列合并
    future_series = pd.Series(future_data, index=future_dates)
    # 为每个分拣中心的预测数据创建一个新的列
    future_df[f'{CS}'] = future_series
# 合并原始及预测数据绘制折线图
future_df.reset_index(inplace=True)
future_df.rename(columns={'index': '日期'}, inplace=True)
melted_pd = future_df.melt(id_vars='日期', var_name='分拣中心', value_name='货量')
print(melted_pd)
data_path = "结果表3.csv"
melted_pd.to_csv(data_path, index=False, encoding='GB2312')
final_df = pd.concat([final_df, melted_pd], ignore_index=True)
CSs = final_df['分拣中心'].unique()[:5]
tem = pd.DataFrame()
for i in CSs:
    # 获取当前分拣中心的数据
    current_center_data = final_df[final_df['分拣中心'] == i]
    # 筛选出12月之前的数据
    pre_december_data = current_center_data[current_center_data['日期'] < '2023-12-01'].copy()
    pre_december_data['颜色'] = 'red'  # 为12月前的数据指定颜色
    # 筛选出12月之后的数据
    post_december_data = current_center_data[current_center_data['日期'] >= '2023-12-01'].copy()
    post_december_data['颜色'] = 'blue'  # 为12月后的数据指定颜色
    # 将两部分数据合并，并添加到tem中
    tem = pd.concat([tem, pre_december_data, post_december_data])
# 创建 FacetGrid 对象
g = sns.FacetGrid(data=tem, row="分拣中心", hue="颜色", height=2.5, aspect=6, sharex=False, sharey=False)
# 绘制线图
g = g.map(sns.lineplot, "日期", "货量", palette="Set1", lw=2)
# 设置x轴标签
plt.xlabel('时间')
# 显示图形
plt.show()