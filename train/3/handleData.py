import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_excel('./handledData.xlsx')
# colors = ['#9999ff', '#2442aa', "blue"]
edu = [1679 / 9586, 1487 / 9586, 1599 / 9586, 1699 / 9586, 1465 / 9586, 1657 / 9586]
labels = ['重要价值用户', '重要挽留用户','重要保持用户','一般挽留用户','一般保持用户','重要发展用户']
explode = [0.2, 0, 0, 0, 0,0]  # 生成数据，用于凸显大专学历人群
colors = sns.color_palette("pastel")
plt.pie(x=edu,  # 绘图数据
        explode=explode,  # 指定饼图某些部分的突出显示，即呈现爆炸式
        labels=labels,  # 添加教育水平标签
        colors=colors,
        autopct='%.2f%%',  # 设置百分比的格式，这里保留两位小数
        pctdistance=0.8,  # 设置百分比标签与圆心的距离
        labeldistance=1.1,  # 设置教育水平标签与圆心的距离
        startangle=180,  # 设置饼图的初始角度
        radius=1.2,  # 设置饼图的半径
        counterclock=False,  # 是否逆时针，这里设置为顺时针方向
        wedgeprops={'linewidth': 1.5, 'edgecolor': '#9999ff'},
        textprops={'fontsize': 10, 'color': 'black'},  # 设置文本标签的属性值
        )

# 添加图标题
# 显示图形
plt.show()