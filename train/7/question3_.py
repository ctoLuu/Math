import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

# 读取数据
df = pd.read_excel("handled_data2.xlsx")

# 预处理数据
df = df[['CustomerID', 'StockCode']]
df.drop_duplicates(inplace=True)

# 创建一个无向图
G = nx.Graph()

# 计算商品和客户的关系
product_customer = df.groupby('StockCode')['CustomerID'].apply(list)

# 为每个商品的顾客对创建边，并加上权重
for customers in product_customer:
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            if G.has_edge(customers[i], customers[j]):
                G[customers[i]][customers[j]]['weight'] += 1
            else:
                G.add_edge(customers[i], customers[j], weight=1)

# 网络分析
print(f"网络中的节点数量: {G.number_of_nodes()}")
print(f"网络中的边数量: {G.number_of_edges()}")

# 计算度分布
degree_distribution = [d for n, d in G.degree()]
plt.hist(degree_distribution, bins=range(1, max(degree_distribution) + 1))
plt.title('Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

# 计算平均聚类系数
average_clustering = nx.average_clustering(G)
print(f"平均聚类系数: {average_clustering}")

# 计算最大连通分量
connected_components = nx.connected_components(G)
largest_component = max(connected_components, key=len)
print(f"最大连通分量的大小: {len(largest_component)}")

# 计算平均路径长度
if nx.is_connected(G):
    avg_path_length = nx.average_shortest_path_length(G)
    print(f"平均路径长度: {avg_path_length}")
else:
    print("图不是连通的，无法计算平均路径长度。")

# 绘制图形
plt.figure(figsize=(8, 8))
nx.draw(G, node_size=10)
plt.show()

# 使用弹簧布局绘制图形
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='grey', node_size=500, font_size=10)
plt.title("Customer Relationship Network")
plt.show()

# 计算社群
partition = community.louvain_communities(G, weight='weight')
print(f"检测到的社群数量: {len(partition)}")

# 计算中心性指标
# 介数中心性
betweenness = nx.betweenness_centrality(G, weight='weight')
print(f"介数中心性: {betweenness}")

# 特征向量中心性
eigenvector = nx.eigenvector_centrality(G, weight='weight')
print(f"特征向量中心性: {eigenvector}")

# 提取社群信息
for i, comm in enumerate(partition):
    print(f"社群 {i + 1}: {comm}")

# 可选：输出每个客户的社区
for node in G.nodes():
    for i, comm in enumerate(partition):
        if node in comm:
            print(f"客户 {node} 属于社群 {i + 1}")
