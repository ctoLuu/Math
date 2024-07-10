import pandas as pd
import numpy as np

# 读取Excel文件
data1 = pd.read_excel('附件1.xlsx')
data2 = pd.read_excel('Q1_1结果(MATLAB读入).xlsx')
data_route1 = pd.read_excel('附件3.xlsx')
data_route2 = pd.read_excel('附件4.xlsx')
data_cluster = pd.read_excel('货量聚类.xlsx')

# 构造训练集
BPSet1 = pd.DataFrame(columns=[*range(10)])
for i, row in data1.iterrows():
    id = row[0]
    # 下面填特征1
    BPSet1.loc[i, 0] = data_cluster.loc[data_cluster[0] == id, 1].values[0] if not data_cluster[0].isnull() else 0

    # 找上游
    upper = data_route1[data_route1[1] == id][0]
    BPSet1.loc[i, 1] = len(upper)  # 特征2
    if len(upper) > 0:
        upper_cluster = data_cluster[data_cluster[0].isin(upper)].iloc[:, 1]
        BPSet1.loc[i, 3] = len(upper_cluster)  # 特征3
        BPSet1.loc[i, 4] = np.mean(upper_cluster)  # 特征4
        BPSet1.loc[i, 6] = np.sum(upper_cluster)  # 特征6
    else:
        BPSet1.loc[i, 3] = 0
        BPSet1.loc[i, 4] = 0


    # 找下游
    lower = data_route1[data_route1[0] == id][0]
    BPSet1.loc[i, 2] = len(lower)  # 特征3
    if len(lower) > 0:
        lower_cluster = data_cluster[data_cluster[0].isin(lower)].iloc[:, 1]
        BPSet1.loc[i, 5] = np.mean(lower_cluster)  # 特征5
        BPSet1.loc[i, 7] = np.sum(lower_cluster)  # 特征7
    else:
        BPSet1.loc[i, 5] = 0
        BPSet1.loc[i, 7] = 0

    BPSet1.loc[i, 8] = row[1]  # 下面填日期

# 构造测试集
BPSet2 = pd.DataFrame(columns=[*range(10)])
for i, row in data2.iterrows():
    id = row[0]
    # 下面填特征1
    BPSet2.loc[i, 0] = data_cluster.loc[data_cluster[0] == id, 1].values[0] if not data_cluster[0].isnull() else 0

    # 找上游
    upper = data_route2[data_route2[1] == id][0]
    BPSet2.loc[i, 1] = len(upper)  # 特征2
    if len(upper) > 0:
        upper_cluster = data_cluster[data_cluster[0].isin(upper)].iloc[:, 1]
        BPSet2.loc[i, 3] = len(upper_cluster)  # 特征3
        BPSet2.loc[i, 4] = np.mean(upper_cluster)  # 特征4
        BPSet2.loc[i, 6] = np.sum(upper_cluster)  # 特征6
    else:
        BPSet2.loc[i, 3] = 0
        BPSet2.loc[i, 4] = 0

    # 找下游
    lower = data_route2[data_route2[0] == id][0]
    BPSet2.loc[i, 2] = len(lower)  # 特征3
    if len(lower) > 0:
        lower_cluster = data_cluster[data_cluster[0].isin(lower)].iloc[:, 1]
        BPSet2.loc[i, 5] = np.mean(lower_cluster)  # 特征5
        BPSet2.loc[i, 7] = np.sum(lower_cluster)  # 特征7
    else:
        BPSet2.loc[i, 5] = 0
        BPSet2.loc[i, 7] = 0

    BPSet2.loc[i, 8] = row[1]  # 下面填日期

# 现在 BPSet1 和 BPSet2 包含了训练集和测试集的数据
