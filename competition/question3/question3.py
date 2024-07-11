import pandas as pd
import numpy as np
from math import floor
import time
import random


class GA(object):
    def __init__(self, lists, time_matrix, slot_num,
                 maxgen=5000,
                 size_pop=100,
                 cross_prob=0.80,
                 pmuta_prob=0.80,
                 multi_pmuta_prob=0.80,
                 all_prob=0.02,
                 select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.multi_prob = multi_pmuta_prob
        self.select_prob = select_prob  # 选择概率
        self.all_prob = all_prob
        self.slot_num = slot_num
        self.lists = lists
        self.time_matrix = time_matrix
        self.package_num = len(self.lists)

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        self.chrom = np.array([0] * self.size_pop * self.package_num * self.slot_num).reshape(size_pop, self.package_num, self.slot_num)
        self.sub_sel = np.array([0] * self.size_pop * self.select_num * self.slot_num).reshape(size_pop, self.select_num, self.slot_num)

        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []
        self.best_matrix = []

    # 随机产生初始化群体函数
    def rand_chrom(self):
        for i in range(self.size_pop):
            for index, list in enumerate(lists):
                left = -1
                right = -1
                leave_number = 0
                for number_index, number in enumerate(list):
                    leave_number = len(list) - number_index
                    left = right
                    right = random.randint(left + 1, self.slot_num - leave_number)
                    self.chrom[i, index, right] = number
            # print(self.chrom[i])
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
        # print(time)
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
        temp = ind_a[r1]
        ind_a[r1] = ind_b[r1]
        ind_b[r1] = temp
        return ind_a, ind_b

    def mutation_sub(self):
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
        find_time.select_sub()
        find_time.cross_sub()
        find_time.mutation_sub()
        find_time.multi_mutation_sub()
        find_time.reins()
        find_time.all_sub()

        for j in range(find_time.size_pop):
            find_time.fitness[j] = find_time.comp_fit(find_time.chrom[j])

        index = find_time.fitness.argmin()
        if (i + 1) % 500 == 0:
            # timestamp = time.time()
            # formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            # print(formatted_time)
            print('第' + str(i+1) + '代后的最短的时间：' + str(find_time.fitness[index]))

        find_time.best_fit.append(find_time.fitness[index])
        find_time.best_matrix.append(find_time.chrom[index])

    best_matrix = find_time.best_matrix.pop()
    print(best_matrix)


if __name__ == "__main__":
    df1 = pd.read_excel('./附件3/Ins3_20_50_10.xlsx', sheet_name=1)
    lists = []
    df1['所需墨盒编号'] = df1['所需墨盒编号'].apply(lambda x: eval(x))
    for index, row in df1.iterrows():
        lists.append(row['所需墨盒编号'])

    df2 = pd.read_excel('./附件3/Ins3_20_50_10.xlsx', sheet_name=2)
    df2.drop(columns=['Unnamed: 0'], inplace=True)
    array = df2.values
    array = np.array(array)
    slot_num = 10
    main(lists, array, slot_num)


