import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_excel("handled_data2.xlsx")

df = df[['CustomerID', 'StockCode']]
df.drop_duplicates(inplace=True)

G = nx.Graph()

for stock_code, group in df.groupby('StockCode'):
    customers = list(group['CustomerID'])
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            G.add_edge(customers[i], customers[j])

print(f"网络中的节点数量: {G.number_of_nodes()}")
print(f"网络中的边数量: {G.number_of_edges()}")

degree_distribution = [d for n, d in G.degree()]
plt.hist(degree_distribution, bins=range(1, max(degree_distribution) + 1))
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

average_clustering = nx.average_clustering(G)
print(f"平均聚类系数: {average_clustering}")

connected_components = nx.connected_components(G)
largest_component = max(connected_components, key=len)
print(f"最大连通分量的大小: {len(largest_component)}")

if nx.is_connected(G):
    avg_path_length = nx.average_shortest_path_length(G)
    print(f"平均路径长度: {avg_path_length}")
else:
    print("图不是连通的，无法计算平均路径长度。")

plt.figure(figsize=(8, 8))
nx.draw(G, node_size=10)
plt.show()

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='grey', node_size=500, font_size=10)
plt.title("Customer Relationship Network")
plt.show()