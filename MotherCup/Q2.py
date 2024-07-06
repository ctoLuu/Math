import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# 设置 matplotlib 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 加载数据
data = pd.read_csv('附件4.csv', encoding='GBK')
data_ini = pd.read_csv('附件3.csv', encoding='GBK')

# 使用pandas的crosstab功能计算不同始发分拣中心到不同到达分拣中心的记录数
distribution_matrix = pd.crosstab(data['始发分拣中心'], data['到达分拣中心'])
distribution_matrix_ini = pd.crosstab(data_ini['始发分拣中心'], data_ini['到达分拣中心'])

print(distribution_matrix)
print(distribution_matrix_ini)
# 创建一个图表，并设置两个子图
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# 定义两组不同的颜色映射
cmap1 = 'Blues'  # 用于第一个热力图的颜色映射
cmap2 = 'Purples'  # 用于第二个热力图的颜色映射

# 在第一个子图上绘制第一个热力图

sns.heatmap(distribution_matrix_ini, annot=True, ax=axs[1], cmap=cmap1, fmt='d')
axs[0].set_title('初始运输记录数')
axs[0].set_xlabel('到达分拣中心')
axs[1].set_ylabel('始发分拣中心')
# 在第二个子图上绘制第二个热力图
sns.heatmap(distribution_matrix, annot=True, ax=axs[0], cmap=cmap2, fmt='d')
axs[1].set_title('当前运输记录数')
axs[1].set_xlabel('到达分拣中心')
axs[1].set_ylabel('始发分拣中心')

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()

# 加载数据
data_new = pd.read_csv('附件3.csv', encoding='GBK')  # 假设编码为GBK，根据之前的文件

# 使用Matplotlib和Seaborn绘制气泡图
plt.figure(figsize=(12, 10))
bubble_plot = sns.scatterplot(data=data_new, x='始发分拣中心', y='到达分拣中心', size='货量', sizes=(50, 3000), legend=None, color='red')
plt.title('始发分拣中心到到达分拣中心的货量分布')
plt.xlabel('始发分拣中心')
plt.ylabel('到达分拣中心')
plt.grid(True)

# 调整x轴和y轴的标签显示以便更容易阅读
bubble_plot.set_xticklabels(bubble_plot.get_xticklabels(), rotation=45)
plt.show()
