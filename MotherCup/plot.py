from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font="simhei",style="whitegrid",font_scale=1.6)
import warnings
warnings.filterwarnings("ignore")
df1 = pd.read_csv('file1.csv', encoding='gbk')
df2 = pd.read_csv('附件2.csv', encoding='gbk')
df3 = pd.read_csv('附件3.csv', encoding='gbk')
df4 = pd.read_csv('附件4.csv', encoding='gbk')

df1['日期'] = pd.to_datetime(df1['日期'])
df2['日期'] = pd.to_datetime(df2['日期'])

df1 = df1.sort_values(by='日期')
df2 = df2.sort_values(by=['日期', '小时'])

CSs = df1['分拣中心'].unique()[-5:]
tem = pd.DataFrame()
for i in CSs:
    tem = pd.concat([tem, df1[df1.分拣中心==i]])
g = sns.FacetGrid(data=tem, row="分拣中心", hue="分拣中心", height=2.5, aspect=6, sharex=False, sharey=False)
g = g.map(sns.lineplot, "日期", "货量", palette="Set1", lw=2)

for index1, i in enumerate(tem["分拣中心"].unique()):
    a = df2['日期'].sort_values().unique()
    ls = []
    for index, i in enumerate(a):
        if index%3 == 0:
            ls.append(i)
        else:
            ls.append('')
plt.xlabel('时间')
plt.show()


