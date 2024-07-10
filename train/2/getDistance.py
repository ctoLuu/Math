import pandas as pd
import numpy as np
from math import cos, asin, sqrt, pi
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']


def haversine(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a))


count = 0
df = pd.read_excel('./数据.xlsx', sheet_name=0)
df.drop(columns=['配送中心', '允许到店时间段'], inplace=True)
df['时间属性'] = df['时间属性'].map({'夜配':0, '日配':1})
df.loc[len(df)] = ['无锡华友', '120.44739', '31.50353', 2]
df['经度'] = df['经度'].astype(float)
df['纬度'] = df['纬度'].astype(float)
time_matrix = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        distance = haversine(df.loc[i, '经度'], df.loc[i, '纬度'],
                             df.loc[j, '经度'], df.loc[j, '纬度']) / 60
        if distance > 8 or distance == 0:
            count += 1
            distance = np.nan
        time_matrix[i][j] = distance
        time_matrix[j][i] = distance
print(f"异常值数量为 {count} ")

print(time_matrix)
time_matrix = pd.DataFrame(time_matrix, columns=df['到达门店简称'])
time_matrix = time_matrix.set_index(df['到达门店简称'])
print(time_matrix)
time_matrix.to_excel('./time_matrix.xlsx', index=True)

x_ticks = df['到达门店简称']
y_ticks = df['到达门店简称']
sns.set_context({"figure.figsize":(15,15)})
ax = sns.heatmap(time_matrix, xticklabels=x_ticks, yticklabels=y_ticks, cmap="YlGnBu", linewidths=1)
ax.set_title('Heatmap for time matrix', fontsize=20)
plt.show()
figure = ax.get_figure()
