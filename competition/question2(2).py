import pandas as pd
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import time

class GA(object):
    # 初始化遗传算法的参数
    def __init__(self, lists, time_matrix, slot_num,
                 maxgen=10000,
                 size_pop=100,
                 cross_prob=0.80,
                 pmuta_prob=0.20,
                 select_prob=0.8):
        self.maxgen = maxgen  # 设置最大迭代次数
        self.size_pop = size_pop  # 设置群体大小
        self.cross_prob = cross_prob  # 设置交叉概率
        self.pmuta_prob = pmuta_prob  # 设置变异概率
        self.select_prob = select_prob  # 设置选择概率
        self.slot_num = slot_num  # 设置插槽数量
        self.lists = lists  # 墨盒编号列表
        self.time_matrix = time_matrix  # 时间矩阵，用于计算适应度
        self.package_num = len(self.lists)  # 包裹数量

        # 根据选择概率计算选择数量，并初始化染色体和子选择数组
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        self.chrom = np.array([0] * self.size_pop * self.package_num * self.slot_num).reshape(
            self.size_pop, self.package_num, self.slot_num)
        self.sub_sel = np.array([0] * self.select_num * self.package_num * self.slot_num).reshape(
            self.select_num, self.package_num, self.slot_num)

        self.fitness = np.zeros(self.size_pop)  # 初始化适应度数组

        # 初始化最佳适应度和最佳矩阵列表
        self.best_fit = []
        self.best_matrix = []

    # 随机产生初始化群体函数
    def rand_chrom(self):
        # 创建一个随机索引数组
        rand_ch = np.array(range(self.slot_num))
        for i in range(self.size_pop):
            for index, list in enumerate(lists):  # 对于每个包裹
                # 打乱随机索引数组
                np.random.shuffle(rand_ch)
                for j in range(self.slot_num):
                    # 将随机索引处的墨盒编号赋值给染色体
                    self.chrom[i, index, rand_ch[j]] = list[j]
                    # 如果已经填满当前包裹的插槽，则跳出循环
                    if j + 1 == len(list):
                        break
            # 计算个体的适应度
            self.fitness[i] = self.comp_fit(self.chrom[i])

    # 计算时间（即当前染色体的适应度）
    def comp_fit(self, matrix):
        time = 0
        # 对于每个包裹从第二个开始
        for i in range(1, self.package_num):
            # 对于每个插槽
            for j in range(self.slot_num):
                # 找到当前插槽中墨盒的上一个包裹编号
                current_i = i
                while (matrix[current_i-1, j] == 0) & (current_i != -1):
                    current_i -= 1
                # 如果找到了上一个包裹编号，则累加时间
                if current_i != -1:
                    time += self.time_matrix[matrix[current_i-1, j] - 1, matrix[i, j] - 1]
        return time

    # 选取子代
    def select_sub(self):
        # 计算适应度的倒数并累加
        fit = 1. / (self.fitness)
        cumsum_fit = np.cumsum(fit)
        # 根据选择概率随机选择子代
        pick = cumsum_fit[-1] / self.select_num * (
            np.random.rand() + np.array(range(int(self.select_num)))
        )
        i, j = 0, 0
        index = []
        # 确定被选择的个体索引
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        # 更新子选择数组
        self.sub_sel = self.chrom[index, :]

    # 交叉，依概率对子代个体进行交叉操作
    def cross_sub(self):
        # 如果选择数量为奇数，从0开始每隔一个选择一个进行交叉
        if self.select_num % 2 == 0:
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num + 1), 2)
        # 对子代个体进行交叉
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i], self.sub_sel[i + 1] = self.inter_cross(self.sub_sel[i], self.sub_sel[i + 1])

    # 交叉函数
    def inter_cross(self, ind_a, ind_b):
        # 随机选择交叉的包裹索引、插槽开始和结束索引
        r1 = np.random.randint(self.package_num)
        r2 = np.random.randint(self.slot_num)
        r3 = np.random.randint(self.slot_num)
        # 确保开始和结束索引不同
        while r2 == r3:
            r3 = np.random.randint(self.slot_num)
        left, right = min(r2, r3), max(r2, r3)
        # 记录原始个体
        ind_a1 = ind_a.copy()
        ind_b1 = ind_b.copy()
        # 执行交叉操作
        for i in range(left, right + 1):
            # 交换两个个体在选定范围内的基因
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[r1, i], ind_b[r1, i] = ind_b1[r1, i], ind_a1[r1, i]
            # 修复非法配置（即同一包裹中出现相同墨盒编号）
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

    # 变异操作
    def mutation_sub(self):
        # 对每个子代个体执行变异操作
        for i in range(int(self.select_num)):
            if np.random.rand() <= self.pmuta_prob:
                r1 = np.random.randint(self.package_num)
                r2 = np.random.randint(self.slot_num)
                r3 = np.random.randint(self.slot_num)
                # 确保变异的插槽索引不同
                while r2 == r3:
                    r3 = np.random.randint(self.slot_num)
                # 交换两个插槽中的墨盒编号
                self.sub_sel[i, r1, [r2, r3]] = self.sub_sel[i, r1, [r3, r2]]

    # 逆转操作
    def reverse_sub(self):
        # 对每个子代个体执行逆转操作
        for i in range(int(self.select_num)):
            r1 = np.random.randint(self.package_num)
            r2 = np.random.randint(self.slot_num)
            r3 = np.random.randint(self.slot_num)
            # 确保逆转的插槽索引不同
            while r2 == r3:
                r3 = np.random.randint(self.slot_num)
            # 记录原始个体
            sel = self.sub_sel[i].copy()
            # 逆转指定范围内的基因序列
            sel[r1, min(r2, r3):max(r2, r3) + 1] = sel[r1, min(r2, r3):max(r2, r3) + 1][::-1]
            # 如果逆转后的个体适应度更低，则接受逆转
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i]):
                self.sub_sel[i] = sel

    # 列交换操作
    def columns_sub(self, epoch):
        for i in range(int(self.select_num)):
            for j in range(max(1, int(epoch / 1000))):
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

    # 重组群体
    def reins(self):
        # 根据适应度从高到低排序，并选择前select_num个个体
        index = np.argsort(self.fitness)[::-1]
        self.chrom[index[:self.select_num]] = self.sub_sel


def main(lists, array, slot_num):
    find_time = GA(lists, array, slot_num)
    find_time.rand_chrom()

    for i in range(find_time.maxgen):  # 执行最大迭代次数
        find_time.select_sub()  # 选择子代
        find_time.cross_sub()  # 交叉操作
        find_time.mutation_sub()  # 变异操作
        find_time.reverse_sub()  # 逆转操作
        find_time.columns_sub(i)  # 列交换操作
        find_time.reins()  # 重组群体

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