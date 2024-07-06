import csv
import numpy as np
import pandas as pd
from scipy.stats import kstest
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy import stats as scs

sns.set(font="simhei",style="whitegrid",font_scale=1.6)
import warnings
warnings.filterwarnings("ignore")
#读取数据
# data_path = '结果表2.csv'
# data = pd.read_csv(data_path, encoding='GB2312')
# data['日期'] = pd.to_datetime(data['日期'])
# data['小时'] = data['小时'].astype(str).str.zfill(2)
# data2_path = "结果表2-final.csv"
# data2 = pd.read_csv(data2_path, encoding='GB2312')
# data2['日期'] = data2['日期'].astype(str)
# data2['小时'] = data2['小时'].astype(str).str.zfill(2)
# data2['日期'] = pd.to_datetime(data2['日期'], errors='coerce')
#
# data['日期时间'] = data['日期'].astype(str) + ' ' + data['小时']
# data['日期时间'] = pd.to_datetime(data['日期时间'])
# data['日期时间'] = pd.to_datetime(data['日期时间'], errors='coerce')
# data.drop(['日期', '小时'], axis=1, inplace=True)
# data.reset_index(drop=True, inplace=True)
# data.rename(columns={'日期时间': '日期小时'}, inplace=True)
# data2['日期时间'] = data2['日期'].astype(str) + ' ' + data2['小时']
# data2['日期时间'] = pd.to_datetime(data2['日期时间'])
# data2['日期时间'] = pd.to_datetime(data2['日期时间'], errors='coerce')
# data2.drop(['日期', '小时'], axis=1, inplace=True)
# data2.reset_index(drop=True, inplace=True)
# data2.rename(columns={'日期时间': '日期小时'}, inplace=True)
# print(data.columns)
# print(data2.columns)
# plt.figure(figsize=(14, 20))  # 设置图形的大小
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 获取分拣中心的唯一值列表，并取第一个分拣中心
# CSs = data['分拣中心'].unique()
#
# # 遍历每个分拣中心
# for CS in CSs:
#     # 从两个数据集中筛选出当前分拣中心的数据
#     current_center_data = data[data['分拣中心'] == CS]
#     current_center_data2 = data2[data2['分拣中心'] == CS]
#
#     # 创建一个新的图形
#     plt.figure()
#
#     # 绘制 data 数据集中当前分拣中心的数据
#     sns.lineplot(x='日期小时', y='货量', data=current_center_data, label='更改路线前数据', linewidth=2)
#
#     # 绘制 data2 数据集中当前分拣中心的数据
#     sns.lineplot(x='日期小时', y='货量', data=current_center_data2, label='更改路线后数据', color='yellow', linewidth=1)
#
#     # 添加图例
#     plt.legend(fontsize='small')
#
#     # 添加 x 轴和 y 轴的标签
#     plt.xlabel('日期小时')
#     plt.ylabel('货量')
#
#     # 旋转 x 轴标签以便于阅读
#     plt.xticks(rotation=30)
#
#     # 设置标题，包含当前分拣中心的名称
#     plt.title(f'分拣中心 {CS} 的货量对比')
#     plt.gcf().autofmt_xdate()
#     # 显示图形
#     plt.show()

# CSs = df['分拣中心'].unique()[:]
# tem = pd.DataFrame()
# for i in CSs:
#     # 获取当前分拣中心的数据
#     current_center_data = df[df['分拣中心'] == i]
#     # 确保'日期时间'列是日期时间类型
#     current_center_data['日期小时'] = pd.to_datetime(current_center_data['日期小时'])
#     # 筛选出12月之前的数据
#     pre_december_data = current_center_data[current_center_data['日期小时'] < datetime(2023, 12, 1)].copy()
#     pre_december_data['颜色'] = 'red'  # 为12月前的数据指定颜色
#     # 筛选出12月之后的数据
#     post_december_data = current_center_data[current_center_data['日期小时'] >= datetime(2023, 12, 1)].copy()
#     post_december_data['颜色'] = 'blue'  # 为12月后的数据指定颜色
#     # 将两部分数据合并，并添加到tem中
#     tem = pd.concat([tem, pre_december_data, post_december_data], ignore_index=True)
# # 创建 FacetGrid 对象
# g = sns.FacetGrid(data=tem, row="分拣中心", hue="颜色", height=2.5, aspect=6, sharex=False, sharey=False)
# # 绘制线图
# g = g.map(sns.lineplot, "日期小时", "货量", palette="Set1", lw=2)
# # 设置x轴标签
# plt.xlabel('时间')
# # 显示图形
# plt.show()

# data3_path = 'file1.csv'
# data3 = pd.read_csv(data3_path, encoding='GB2312')
# data3['日期'] = pd.to_datetime(data3['日期'])
# data4_path = "结果表3_.xlsx"
# data4 = pd.read_excel(data4_path)
# data4['日期'] = data4['日期'].astype(str)
# data4['日期'] = pd.to_datetime(data4['日期'], errors='coerce')
# # 合并原始及预测数据
# final_df = pd.concat([data3, data4], ignore_index=True)
# CSs = final_df['分拣中心'].unique()[:]
# tem = pd.DataFrame()
# for i in CSs:
#     # 获取当前分拣中心的数据
#     current_center_data = final_df[final_df['分拣中心'] == i]
#     # 筛选出12月之前的数据
#     pre_december_data = current_center_data[current_center_data['日期'] < '2023-12-01'].copy()
#     pre_december_data['颜色'] = 'red'  # 为12月前的数据指定颜色
#     # 筛选出12月之后的数据
#     post_december_data = current_center_data[current_center_data['日期'] >= '2023-12-01'].copy()
#     post_december_data['颜色'] = 'blue'  # 为12月后的数据指定颜色
#     # 将两部分数据合并，并添加到tem中
#     tem = pd.concat([tem, pre_december_data, post_december_data])
# # 创建 FacetGrid 对象
# g = sns.FacetGrid(data=tem, row="分拣中心", hue="颜色", height=2.5, aspect=6, sharex=False, sharey=False)
# # 绘制线图
# g = g.map(sns.lineplot, "日期", "货量", palette="Set1", lw=2)
# # 设置x轴标签
# plt.xlabel('时间')
# # 显示图形
# plt.show()
data_path = "附件2.csv"
data = pd.read_csv(data_path, encoding='GB2312')
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


CSs = df['分拣中心'].unique()[-11:-6]
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