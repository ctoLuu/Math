import pandas as pd  # 导入Pandas库，用于数据处理
import numpy as np    # 导入NumPy库，用于数组操作
from math import floor  # 导入floor函数，用于向下取整
import time           # 导入time库，用于时间相关操作
import random         # 导入random库，用于生成随机数

class GA(object):
    # 初始化遗传算法的实例
    def __init__(self, lists, time_matrix, slot_num,
                 maxgen=10000,
                 size_pop=400,
                 cross_prob=0.80,
                 pmuta_prob=0.40,
                 shuffle_pmuta_prob=0.1,
                 multi_pmuta_prob=0.80,
                 all_prob=0.02,
                 select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 种群大小
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 个体变异概率
        self.shuffle_pmuta_prob = shuffle_pmuta_prob  # 洗牌变异概率
        self.multi_prob = multi_pmuta_prob  # 多重变异概率
        self.select_prob = select_prob  # 选择概率
        self.all_prob = all_prob  # 整体变异概率
        self.slot_num = slot_num  # 时间槽数量
        self.lists = lists  # 每个任务所需的包裹列表
        self.time_matrix = time_matrix  # 时间矩阵，表示不同包裹在不同时间槽的耗时
        self.package_num = len(self.lists)  # 包裹总数

        # 根据选择概率计算选择的个体数量
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        # 初始化染色体数组，用于存储种群中每个个体的基因
        self.chrom = np.array([0] * self.size_pop * self.package_num * (self.slot_num+1)).reshape(self.size_pop, self.package_num, self.slot_num+1)
        # 初始化选择后的子代数组
        self.sub_sel = np.array([0] * self.size_pop * self.select_num * (self.slot_num+1)).reshape(self.size_pop, self.select_num, self.slot_num+1)
        # 初始化适应度数组
        self.fitness = np.zeros(self.size_pop)

        # 初始化记录最佳适应度和最佳矩阵的列表
        self.best_fit = []
        self.best_matrix = []

    # 随机产生初始化群体函数
    def rand_chrom(self):
        # 遍历每个个体
        for i in range(self.size_pop):
            # 对每个个体的每个包裹进行处理
            for index, list in enumerate(self.lists):
                # 初始化变量
                left = -1
                right = -1
                leave_number = 0
                # 设置最后一个时间槽的包裹编号
                self.chrom[i, index, self.slot_num] = index
                # 对包裹中的每个时间点进行处理
                for number_index, number in enumerate(list):
                    # 更新剩余包裹数量
                    leave_number = len(list) - number_index
                    # 随机生成当前时间点的左右边界
                    left = right
                    right = random.randint(left + 1, self.slot_num - leave_number)
                    # 设置当前时间点的包裹编号
                    self.chrom[i, index, right] = number
            # 打乱个体染色体的顺序
            np.random.shuffle(self.chrom[i])
            # 计算当前个体的适应度
            self.fitness[i] = self.comp_fit(self.chrom[i])

    # 计算时间（即当前染色体的适应度）
    def comp_fit(self, matrix):
        # 初始化总时间
        time = 0
        # 对每个任务进行处理
        for i in range(1, self.package_num):
            # 对每个时间槽进行处理
            for j in range(self.slot_num):
                # 找到当前任务之前的第一个任务
                current_i = i
                while (matrix[current_i-1, j] == 0) & (current_i != -1):
                    current_i -= 1
                # 如果找到，则累加所需时间
                if current_i != -1:
                    time += self.time_matrix[matrix[current_i-1, j] - 1, matrix[i, j] - 1]
        return time

    # 选取子代
    def select_sub(self):
        # 计算适应度的倒数
        fit = 1. / (self.fitness)
        # 计算适应度的累积和
        cumsum_fit = np.cumsum(fit)
        # 根据选择概率选择子代
        pick = cumsum_fit[-1] / self.select_num * (np.random.rand() + np.array(range(int(self.select_num))))
        # 初始化选择后的个体索引列表
        index = []
        # 遍历种群，选择子代
        i, j = 0, 0
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        # 更新子代数组
        self.sub_sel = self.chrom[index, :]

    # 交叉，依概率对子代个体进行交叉操作
    def cross_sub(self):
        # 根据子代数量确定交叉的配对数量
        if self.select_num % 2 == 0:
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num + 1), 2)
        # 对每对子代进行交叉
        for i in num:
            if self.cross_prob >= np.random.rand():
                # 执行交叉操作
                self.sub_sel[i], self.sub_sel[i + 1] = self.inter_cross(self.sub_sel[i], self.sub_sel[i + 1])

    def inter_cross(self, ind_a, ind_b):
        # 选择一个包裹作为交叉点
        r1 = np.random.randint(self.package_num)
        flag = ind_a[r1, self.slot_num]
        r2 = 0
        # 在第二个个体中找到交叉点对应的包裹
        for i in range(self.package_num):
            if ind_b[i, self.slot_num] == flag:
                r2 = i
                break
        # 交换两个个体中交叉点包裹的位置
        temp = ind_a[r1]
        ind_a[r1] = ind_b[r2]
        ind_b[r2] = temp
        return ind_a, ind_b

    def mutation_sub(self):
        for i in range(int(self.select_num)):
            if np.random.rand() <= self.shuffle_pmuta_prob:
                r1 = np.random.randint(self.package_num)
                r2 = np.random.randint(self.package_num)
                while r2 == r1:
                    r2 = np.random.randint(self.package_num)
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i, [r2, r1]]

        for i in range(int(self.select_num)):
            if np.random.rand() <= self.pmuta_prob:
                r1 = np.random.randint(self.package_num)
                r2 = np.random.randint(self.slot_num)
                while self.sub_sel[i, r1, r2] == 0:
                    r2 = np.random.randint(self.slot_num)
                left = r2
                right = r2
                if left != 0:
                    while self.sub_sel[i, r1, left - 1] == 0:
                        left -= 1
                        if left == 0:
                            break
                if right != self.slot_num - 1:
                    while self.sub_sel[i, r1, right + 1] == 0:
                        right += 1
                        if right == self.slot_num - 1:
                            break
                r3 = random.randint(left, right)
                if left != right:
                    while r2 == r3:
                        r3 = random.randint(left, right)
                self.sub_sel[i, r1, [r2, r3]] = self.sub_sel[i, r1, [r3, r2]]

    def multi_mutation_sub(self):
        for i in range(int(self.select_num)):
            for j in range(2):
                if np.random.rand() <= self.multi_prob:
                    r1 = np.random.randint(self.package_num)
                    r2 = np.random.randint(self.slot_num)
                    while self.sub_sel[i, r1, r2] == 0:
                        r2 = np.random.randint(self.slot_num)
                    left = r2
                    right = r2
                    if left != 0:
                        while self.sub_sel[i, r1, left - 1] == 0:
                            left -= 1
                            if left == 0:
                                break
                    if right != self.slot_num - 1:
                        while self.sub_sel[i, r1, right + 1] == 0:
                            right += 1
                            if right == self.slot_num - 1:
                                break
                    r3 = random.randint(left, right)
                    if left != right:
                        while r2 == r3:
                            r3 = random.randint(left, right)
                    self.sub_sel[i, r1, [r2, r3]] = self.sub_sel[i, r1, [r3, r2]]

    def all_sub(self):
        for i in range(int(self.select_num)):
            if np.random.rand() <= self.all_prob:
                for j in range(int(self.package_num)):
                    r2 = np.random.randint(self.slot_num)
                    while self.sub_sel[i, j, r2] == 0:
                        r2 = np.random.randint(self.slot_num)
                    left = r2
                    right = r2
                    if left != 0:
                        while self.sub_sel[i, j, left - 1] == 0:
                            left -= 1
                            if left == 0:
                                break
                    if right != self.slot_num - 1:
                        while self.sub_sel[i, j, right + 1] == 0:
                            right += 1
                            if right == self.slot_num - 1:
                                break
                    r3 = random.randint(left, right)
                    if left != right:
                        while r2 == r3:
                            r3 = random.randint(left, right)
                    self.sub_sel[i, j, [r2, r3]] = self.sub_sel[i, j, [r3, r2]]

    def reins(self):
        index = np.argsort(self.fitness)[::-1]
        self.chrom[index[:self.select_num]] = self.sub_sel


def main(lists, array, slot_num):
    find_time = GA(lists, array, slot_num)
    find_time.rand_chrom()

    for i in range(find_time.maxgen):
        # 选择
        find_time.select_sub()
        # 交叉
        find_time.cross_sub()
        # 变异
        find_time.mutation_sub()
        # 多重变异
        find_time.multi_mutation_sub()
        # 全部变异
        find_time.all_sub()
        # 重新插入
        find_time.reins()

        # 计算当前种群的适应度
        for j in range(find_time.size_pop):
            find_time.fitness[j] = find_time.comp_fit(find_time.chrom[j])

        # 记录每代的最优适应度和染色体
        index = find_time.fitness.argmin()
        if (i + 1) % 5 == 0:
            # 打印每500代的最优适应度
            print('第' + str(i + 1) + '代后的最短的时间：' + str(find_time.fitness[index]))
        if (i + 1) % 200 == 0:
            # 打印每2000代的最优染色体
            best_matrix = find_time.best_matrix.pop()
            print(best_matrix)

        find_time.best_fit.append(find_time.fitness[index])
        find_time.best_matrix.append(find_time.chrom[index])

    # 输出最终的最优染色体
    best_matrix = find_time.best_matrix.pop()
    print(best_matrix)


if __name__ == "__main__":
    df1 = pd.read_excel('./附件4/Ins4_20_40_10.xlsx', sheet_name=1)
    lists = []
    df1['所需墨盒编号'] = df1['所需墨盒编号'].apply(lambda x: eval(x))
    for index, row in df1.iterrows():
        lists.append(row['所需墨盒编号'])

    df2 = pd.read_excel('./附件4/Ins4_20_40_10.xlsx', sheet_name=2)
    df2.drop(columns=['Unnamed: 0'], inplace=True)
    array = df2.values
    array = np.array(array)
    for i in range(len(array)):
        for j in range(len(array)):
            if i == j:
                continue
            else:
                time = random.randint(-2, 2)
                array[i, j] += time
    slot_num = 10
    main(lists, array, slot_num)


