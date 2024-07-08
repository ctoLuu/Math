import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import nonzero, array
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, rand_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

df = pd.read_excel('./question4.xlsx')  # 设置要读取的数据集
df.dropna(inplace=True)
# print(df)

columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns)][1:]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
print(features)
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
# scaler = StandardScaler()
# dataset = scaler.fit_transform(dataset)
print(dataset)
attributes = len(df.columns) - 1  # 属性数量（数据集维度）
original_labels = list(df[columns[-1]])  # 原始标签



def initialize_centroids(data, k):
    # 从数据集中随机选择k个点作为初始质心
    centers = data[np.random.choice(data.shape[0], k, replace=False)]
    return centers


def get_clusters(data, centroids):
    # 计算数据点与质心之间的距离，并将数据点分配给最近的质心
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    print(cluster_labels)
    return cluster_labels


def update_centroids(data, cluster_labels, k):
    # 计算每个簇的新质心，即簇内数据点的均值
    new_centroids = np.array([data[cluster_labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


def k_means(data, k, T, epsilon):
    start = time.time()  # 开始时间，计时
    # 初始化质心
    centroids = initialize_centroids(data, k)
    t = 0
    list = [[], [], [], [], [], []]
    while t <= T:
        # 分配簇
        cluster_labels = get_clusters(data, centroids)

        # 更新质心
        new_centroids = update_centroids(data, cluster_labels, k)

        # 检查收敛条件
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            for i in range(len(cluster_labels)):
                if cluster_labels[i] == 0:
                    list[0].append(i)
                elif cluster_labels[i] == 1:
                    list[1].append(i)
                elif cluster_labels[i] == 2:
                    list[2].append(i)
                elif cluster_labels[i] == 3:
                    list[3].append(i)
                elif cluster_labels[i] == 4:
                    list[4].append(i)
                elif cluster_labels[i] == 5:
                    list[5].append(i)
            # centroids = scalar.inverse_transform(centroids)
            print(centroids)
            break
        centroids = new_centroids
        # centroids = scalar.inverse_transform(centroids)
        print("第", t, "次迭代")
        t += 1
    print("用时：{0}".format(time.time() - start))
    return cluster_labels, centroids, list


# 计算聚类指标
def clustering_indicators(labels_true, labels_pred):
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])  # 如果数据集的标签为文本类型，把文本标签转换为数字标签
    f_measure = f1_score(labels_true, labels_pred, average='macro')  # F值
    accuracy = accuracy_score(labels_true, labels_pred)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    ARI = adjusted_rand_score(labels_true, labels_pred)
    return f_measure, accuracy, normalized_mutual_information, rand_index, ARI


# 绘制聚类结果散点图
def draw_cluster(dataset, centers, labels):
    center_array = array(centers)
    if attributes > 2:
        dataset = PCA(n_components=2).fit_transform(dataset)  # 如果属性数量大于2，降维
        center_array = PCA(n_components=2).fit_transform(center_array)  # 如果属性数量大于2，降维
    else:
        dataset = array(dataset)
    # 做散点图
    label = array(labels)
    plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c='black', s=7)  # 原图
    # plt.show()
    colors = np.array(
        ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000",
         "#800080", "#008080", "#444444", "#FFD700", "#008080"])
    # 循换打印k个簇，每个簇使用不同的颜色
    for i in range(k):
        plt.scatter(dataset[nonzero(label == i), 0], dataset[nonzero(label == i), 1], c=colors[i], s=7, marker='o')
    # plt.scatter(center_array[:, 0], center_array[:, 1], marker='x', color='m', s=30)  # 聚类中心
    plt.show()

if __name__ == "__main__":
    k = 6  # 聚类簇数
    T = 1000  # 最大迭代数
    n = len(dataset)  # 样本数
    epsilon = 1e-5
    # 预测全部数据
    labels, centers, list = k_means(np.array(dataset), k, T, epsilon)
    # print(labels)
    # F_measure, ACC, NMI, RI, ARI = clustering_indicators(original_labels, labels)  # 计算聚类指标
    # print("F_measure:", F_measure, "ACC:", ACC, "NMI", NMI, "RI", RI, "ARI", ARI)
    # print(membership)
    print(centers)
    print(list[0])
    print(list[1])
    print(list[2])
    print(list[3])
    print(list[4])
    print(list[5])
    plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号的处理
    plt.axes(aspect='equal')  # 将横、纵坐标轴标准化处理，确保饼图是一个正圆，否则为椭圆

    # length = len(dataset)
    # print(length)
    # edu = [len(list[0]) / length, len(list[1]) / length, len(list[2]) / length, len(list[3]) / length, len(list[4]) / length]
    # labels = ['batter', 'great', 'middle', 'bad', 'worse']
    # explode = [0, 0.1, 0, 0, 0]  # 生成数据，用于凸显大专学历人群
    # colors = ['#9999ff', '#ff9999', '#7777aa', '#2442aa', '#dd5555']  # 自定义颜色
    #
    # plt.pie(x=edu,  # 绘图数据
    #         explode=explode,  # 指定饼图某些部分的突出显示，即呈现爆炸式
    #         labels=labels,  # 添加教育水平标签
    #         colors=colors,
    #         autopct='%.2f%%',  # 设置百分比的格式，这里保留两位小数
    #         pctdistance=0.8,  # 设置百分比标签与圆心的距离
    #         labeldistance=1.1,  # 设置教育水平标签与圆心的距离
    #         startangle=180,  # 设置饼图的初始角度
    #         radius=1.2,  # 设置饼图的半径
    #         counterclock=False,  # 是否逆时针，这里设置为顺时针方向
    #         wedgeprops={'linewidth': 1.5, 'edgecolor': 'green'},  # 设置饼图内外边界的属性值
    #         textprops={'fontsize': 10, 'color': 'black'},  # 设置文本标签的属性值
    #         )
    #
    # # 添加图标题
    # # 显示图形
    # plt.show()
    print(dataset)
    draw_cluster(dataset, centers, labels=labels)

