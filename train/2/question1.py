import pandas as pd
import numpy as np

df = pd.read_excel('./数据.xlsx', sheet_name=0)
df.drop(columns=['配送中心', '允许到店时间段'], inplace=True)
df['时间属性'] = df['时间属性'].map({'夜配':0, '日配':1})
print(df)
time_matrix = pd.read_excel('./time_matrix.xlsx')
print(time_matrix)

class Gene_TSP(object):
    def __init__(self, data, maxgen=1000, size_pop=100, cross_prob=0.80, pmuta_prob=0.02, select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.data = data
        self.