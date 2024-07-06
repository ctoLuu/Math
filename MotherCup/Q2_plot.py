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
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import os
from scipy.interpolate import interp1d


sns.set(font="simhei",style="whitegrid",font_scale=1.6)
import warnings
warnings.filterwarnings("ignore")

data_path = 'file1.csv'
data = pd.read_csv(data_path, encoding='gbk')
data['日期'] = pd.to_datetime(data['日期'])

data2_path = "结果表3.csv"
data2 = pd.read_csv(data2_path, encoding='utf-8-sig')

data2['预测日期'] = data2['预测日期'].astype(str)
data2.rename(columns={'预测日期': '日期'}, inplace=True)
data2.rename(columns={'预测货量': '货量'}, inplace=True)
data2['日期'] = pd.to_datetime(data2['日期'])
final_df = pd.concat([data, data2], ignore_index=True)

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