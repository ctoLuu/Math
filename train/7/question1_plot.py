import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_excel('handled_data1.xlsx')
list2 = df['StockCode'].unique()
index_map = {value: index for index, value in enumerate(list2)}
c = [[] for _ in range(3940)]
for row, index in df.iterrows():
    target_index = index_map.get(df.loc[row, 'StockCode'])
    c[target_index].append(row)

num = [0 for _ in range(3940)]
for i in range(3940):
    current = c[i]
    for j in c[i]:
        num[i] += 1
# 饼图绘制
plt.figure(figsize=(8, 8))
plt.pie(num)
plt.title('Pie Chart of List Data')
plt.show()