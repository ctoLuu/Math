import pandas as pd
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import time


class GA(object):
    def __init__(self, lists, time_matrix, slot_num,
                 maxgen=10000,
                 size_pop=200,
                 cross_prob=0.80,
                 pmuta_prob=0.20,
                 select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率
        self.slot_num = slot_num
        self.lists = lists
        self.time_matrix = time_matrix
        self.package_num = len(self.lists)

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        self.chrom = np.array([0] * self.size_pop * self.package_num * self.slot_num).reshape(size_pop, self.package_num, self.slot_num)
        self.sub_sel = np.array([0] * self.select_num * self.package_num * self.slot_num).reshape(self.select_num, self.package_num, self.slot_num)

        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []
        self.best_matrix = []

    # 随机产生初始化群体函数
    def rand_chrom(self):
        rand_ch = np.array(range(self.slot_num))
        for i in range(self.size_pop):
            for index, list in enumerate(lists):
                np.random.shuffle(rand_ch)
                for j in range(self.slot_num):
                    self.chrom[i, index, rand_ch[j]] = list[j]
                    if j + 1 == len(list):
                        break
            self.fitness[i] = self.comp_fit(self.chrom[i])

    # 计算时间（即当前染色体的适应度）
    def comp_fit(self, matrix):
        time = 0
        for i in range(1, self.package_num):
            for j in range(self.slot_num):
                current_i = i
                while (matrix[current_i-1, j] == 0) & (current_i != -1):
                    current_i -= 1
                if current_i != -1:
                    time += self.time_matrix[matrix[current_i-1, j] - 1, matrix[i, j] - 1]
        return time

    # 选取子代
    def select_sub(self):
        fit = 1. / (self.fitness)
        cumsum_fit = np.cumsum(fit)
        pick = cumsum_fit[-1] / self.select_num * (
            np.random.rand() + np.array(range(int(self.select_num)))
        )
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]

    # 交叉，依概率对子代个体进行交叉操作
    def cross_sub(self):
        if self.select_num % 2 == 0:
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num + 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i], self.sub_sel[i + 1] = self.inter_cross(self.sub_sel[i], self.sub_sel[i + 1])

    def inter_cross(self, ind_a, ind_b):
        r1 = np.random.randint(self.package_num)
        r2 = np.random.randint(self.slot_num)
        r3 = np.random.randint(self.slot_num)
        while r2 == r3:
            r3 = np.random.randint(self.slot_num)
        left, right = min(r2, r3), max(r2, r3)
        ind_a1 = ind_a.copy()
        ind_b1 = ind_b.copy()
        for i in range(left, right + 1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[r1, i] = ind_b1[r1, i]
            ind_b[r1, i] = ind_a1[r1, i]
            x = np.argwhere(ind_a[r1] == ind_a[r1, i])
            y = np.argwhere(ind_b[r1] == ind_b[r1, i])
            if len(x) == 2:
                ind_a[r1, x[x != i]] = ind_a2[r1, i]
            if len(y) == 2:
                ind_b[r1, y[y != i]] = ind_b2[r1, i]
            if len(x) > 2:
                ind_a[r1] = ind_a2[r1]
            if len(y) > 2:
                ind_b[r1] = ind_b2[r1]
        return ind_a, ind_b

    def mutation_sub(self):
        for i in range(int(self.select_num)):
            if np.random.rand() <= self.pmuta_prob:
                r1 = np.random.randint(self.package_num)
                r2 = np.random.randint(self.slot_num)
                r3 = np.random.randint(self.slot_num)
                while r2 == r3:
                    r3 = np.random.randint(self.slot_num)
                self.sub_sel[i, r1, [r2, r3]] = self.sub_sel[i, r1, [r3, r2]]


    def reverse_sub(self):
        for i in range(int(self.select_num)):
            r1 = np.random.randint(self.package_num)
            r2 = np.random.randint(self.slot_num)
            r3 = np.random.randint(self.slot_num)
            while r2 == r3:
                r3 = np.random.randint(self.slot_num)
            left, right = min(r2, r3), max(r2, r3)
            sel = self.sub_sel[i].copy()
            sel[r1, left:right + 1] = self.sub_sel[i, r1, left:right + 1][::-1]
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i, :]):
                self.sub_sel[i] = sel


    def columns_sub(self, epoch):
        for i in range(int(self.select_num)):
            for j in range(max(1, int(epoch / 100))):
                r1 = np.random.randint(self.package_num)
                r2 = np.random.randint(self.slot_num)
                r3 = np.random.randint(self.slot_num)
                while r2 == r3:
                    r3 = np.random.randint(self.slot_num)
                sel = self.sub_sel[i].copy()
                for row in range(r1, self.package_num):
                    sel[row, [r2, r3]] = sel[row, [r3, r2]]
                if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i]):
                    self.sub_sel[i] = sel

    def reins(self):
        index = np.argsort(self.fitness)[::-1]
        self.chrom[index[:self.select_num]] = self.sub_sel


def main(lists, array, slot_num):
    find_time = GA(lists, array, slot_num)
    find_time.rand_chrom()

    for i in range(find_time.maxgen):
        find_time.select_sub()
        find_time.cross_sub()
        find_time.mutation_sub()
        find_time.reverse_sub()
        find_time.columns_sub(i)
        find_time.reins()

        for j in range(find_time.size_pop):
            find_time.fitness[j] = find_time.comp_fit(find_time.chrom[j])

        index = find_time.fitness.argmin()
        if (i + 1) % 1000 == 0:
            # timestamp = time.time()
            # formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            # print(formatted_time)
            print('第' + str(i+1) + '代后的最短的时间：' + str(find_time.fitness[index]))

        if (i + 1) % 1000 == 0:
            best_matrix = find_time.best_matrix.pop()
            print(best_matrix)

        find_time.best_fit.append(find_time.fitness[index])
        find_time.best_matrix.append(find_time.chrom[index])


    # 程序结束前绘制图像
    plt.figure(figsize=(10, 5))  # 设置图像大小
    generations = list(range(10000))
    # 绘制最佳适应度曲线，但只显示每1000个epoch的点
    plt.plot(generations, find_time.best_fit, label='Best Fitness', marker='o', linewidth=0.75)  # 使用圆点标记每个数据点

    # 每隔1000个epoch显示一个x轴刻度
    plt.xticks(generations[::500], [str(gen) for gen in generations[::500]])

    plt.xlabel('Generation (Every 1000 epochs)')  # 设置x轴标签
    plt.ylabel('Best Fitness')  # 设置y轴标签
    plt.title('Best Fitness Over Generations')  # 设置图像标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图像

    best_matrix = find_time.best_matrix.pop()
    print(best_matrix)


if __name__ == "__main__":
    df1 = pd.read_excel('./附件2/Ins5_30_60_10.xlsx', sheet_name=1)
    lists = []
    df1['所需墨盒编号'] = df1['所需墨盒编号'].apply(lambda x: eval(x))
    for index, row in df1.iterrows():
        lists.append(row['所需墨盒编号'])

    df2 = pd.read_excel('./附件2/Ins5_30_60_10.xlsx', sheet_name=2)
    df2.drop(columns=['Unnamed: 0'], inplace=True)
    array = df2.values
    array = np.array(array)

    slot_num = 10
    main(lists, array, slot_num)


