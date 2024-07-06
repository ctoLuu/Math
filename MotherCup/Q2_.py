import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取Excel文件
data = pd.read_excel('附件3.xlsx')

# 创建图
G = nx.Graph()

# 添加边和节点，并为每个节点设置初始的'weight'属性
for _, row in data.iterrows():
    src_node = row['始发分拣中心']
    dest_node = row['到达分拣中心']
    weight = row['货量']
    G.add_edge(src_node, dest_node, weight=weight)
    # 如果节点不在图中，添加节点并设置'weight'属性为0
    if src_node not in G:
        G.add_node(src_node, weight=0)
    if dest_node not in G:
        G.add_node(dest_node, weight=0)

# 更新节点的'weight'属性，累加每个节点的流入和流出货量
for src, dest, attrs in G.edges(data=True):
    weight = attrs['weight']
    # 如果节点有'weight'属性，更新它，否则设置为0
    G.nodes[src]['weight'] = G.nodes[src].get('weight', 0) + weight
    G.nodes[dest]['weight'] = G.nodes[dest].get('weight', 0) + weight

# 计算最大权重
max_weight = max(G.nodes[node].get('weight', 0) for node in G)

# 根据最大权重调整节点大小
node_size = [G.nodes[node].get('weight', 0) * 100 / max_weight for node in G]

# 绘制图形
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=1.5)

# 绘制边，根据边的权重设置边宽
edge_width = [edge[2]['weight'] / max_weight * 5 for edge in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7)

# 绘制节点
# 使用scatter函数来绘制节点，并设置节点的大小和颜色
plt.scatter([x for x, y in pos.values()], [y for x, y in pos.items()], s=node_size, color='lightblue', alpha=0.7)

# 绘制节点标签
nx.draw_networkx_labels(G, pos)

# 设置标题和坐标轴
plt.title('物流运输线路拓扑关系图')
plt.axis('off')

# 显示图形
plt.show()