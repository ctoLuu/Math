# 计算第一问第一种情况下第二部分子染色体2的遗传算法

import pandas as pd
import numpy as np
from math import floor
import time
import matplotlib.pyplot as plt
import random
import copy

class GA(object):
    def __init__(self, count, count2, S, price, output, cost, prev,
                 maxgen=2000,
                 size_pop=500,
                 cross_prob=0.80,
                 pmuta_prob=0.02,
                 select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 种群数量
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.num = 26 * 7  # 染色体总数
        self.count = count  # 预测销量
        self.count2 = count2  # 第二种预测销量
        self.S = S  # 地块面积
        self.price = price  # 单价
        self.output = output  # 亩产量
        self.cost = cost  # 种植成本

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)  # 选择个体数
        self.chrom = []  # 保存染色体
        self.sub_sel = []  # 选择的子代
        self.fitness = np.zeros(self.size_pop)  # 适应度

        self.best_fit = []  # 最优适应度
        self.best_path = []  # 最优路径

        self.prev = prev  # 上一作物编号

    # 随机生成染色体
    def get_rand_ch(self, index):
        seed = np.zeros(15)
        available_indices = list(set(range(15)))
        num_to_generate = np.random.choice([1, 2])  # 随机选择生成一个或两个数

        max_sum = self.S[index]  # 最大可能和
        select_max = max_sum / 3 * 2
        min_sum = int(max_sum / 3)  # 最小可能和

        if num_to_generate == 1:
            idx1 = np.random.choice(available_indices)
            seed[idx1] = max_sum  # 生成一个数
        else:
            idx1, idx2 = np.random.choice(available_indices, 2, replace=False)  # 生成两个数
            value1 = np.random.randint(min_sum, select_max + 1)
            value2 = max_sum - value1
            seed[idx1] = value1
            seed[idx2] = value2
        return seed

    # 随机初始化种群
    def rand_chrom(self):
        for i in range(self.size_pop):
            rand_ch = [np.zeros((7, 15)) for _ in range(26)]  # 初始化种群
            for index, seed in enumerate(rand_ch):
                prev_used_indices = set([self.prev[index] - 1])  # 已使用索引
                for row in seed:  # 遍历种群的每一行
                    available_indices = list(set(range(15)) - prev_used_indices)  # 当前行可用索引
                    num_to_generate = np.random.choice([1, 2])  # 随机生成1或2个数

                    max_sum = self.S[index]
                    select_max = max_sum / 3 * 2
                    min_sum = int(max_sum / 3)

                    if num_to_generate == 1:
                        idx1 = np.random.choice(available_indices)
                        row[idx1] = max_sum
                        prev_used_indices = {idx1}  # 更新已使用索引
                    else:
                        if len(available_indices) >= 2:  # 确保可用索引至少有两个
                            idx1, idx2 = np.random.choice(available_indices, 2, replace=False)
                            value1 = np.random.randint(min_sum, select_max + 1)
                            value2 = max_sum - value1
                            row[idx1] = value1
                            row[idx2] = value2
                            prev_used_indices = {idx1, idx2}
                        else:
                            idx1 = np.random.choice(available_indices)
                            row[idx1] = max_sum
                            prev_used_indices = {idx1}
            self.get_chrom(rand_ch)  # 生成染色体并计算适应度
            self.fitness[i] = self.comp_fit(rand_ch)
            self.chrom.append(rand_ch)
        self.check()  # 检查种群

    # 生成染色体
    def get_chrom(self, rand_ch):
        for index, seed in enumerate(rand_ch):
            try:
                for j in range(0, 5, 1):  # 遍历每三行
                    sum_first_five_cols = np.sum(seed[j:j + 3, :5])  # 计算前三列和
                    while sum_first_five_cols < self.S[index]:  # 检查条件
                        non_zero_indices = np.argwhere(seed[j:j + 3, 5:] != 0)  # 查找非零元素索引
                        if len(non_zero_indices) == 0:
                            raise ValueError("没有找到非零元素来修改")  # 如果没有非零元素，抛出异常

                        row_to_modify, col_to_modify = non_zero_indices[np.random.choice(len(non_zero_indices))]  # 随机选择非零元素
                        row_to_modify += j
                        col_to_modify += 5

                        non_zero_indices_prev_row = np.nonzero(seed[row_to_modify - 1][:5])[0] if row_to_modify - 1 >= 0 else set()
                        non_zero_indices_cur_row = np.nonzero(seed[row_to_modify][:5])[0]
                        non_zero_indices_next_row = np.nonzero(seed[row_to_modify][:5])[0]
                        if row_to_modify + 1 < 7:
                            non_zero_indices_next_row = np.nonzero(seed[row_to_modify + 1][:5])[0]

                        available_positions = list(set(range(5)) - set(non_zero_indices_prev_row) - set(non_zero_indices_cur_row) - set(non_zero_indices_next_row))
                        if not available_positions:
                            raise ValueError("没有可用位置来移动该值")  # 如果没有可用位置，抛出异常

                        new_col_index = np.random.choice(available_positions)
                        if row_to_modify + 1 < len(seed):
                            non_zero_indices_next_row = np.nonzero(seed[row_to_modify + 1][:5])[0]
                            if new_col_index in non_zero_indices_next_row:
                                available_positions.remove(new_col_index)
                                if not available_positions:
                                    raise ValueError("没有可用位置来移动该值")
                                new_col_index = np.random.choice(available_positions)

                        seed[row_to_modify, new_col_index] = seed[row_to_modify, col_to_modify]
                        seed[row_to_modify, col_to_modify] = 0

                        sum_first_five_cols = np.sum(seed[j:j + 3, :5])  # 修改后的重新计算和

            except ValueError as e:
                print(f"在索引 {index} 处的 seed 发生错误: {e}")  # 捕获错误时输出当前种群状态
                print("当前种群状态:")
                print(seed)
                raise  # 重新引发异常

        return rand_ch

    # 计算适应度
    def comp_fit(self, array):
        self.get_chrom(array)
        fitness = 0
        for i in range(7):
            column_sums = [0] * 15
            cost_sums = [0] * 15
            for index, seed in enumerate(array):
                non_zero_indices_cur_row = np.nonzero(seed[0])[0]
                if self.prev[index] - 1 in non_zero_indices_cur_row:
                    return 0
                if index < 6:
                    for j in range(15):
                        column_sums[j] += seed[i][j] * output[j]
                        cost_sums[j] += seed[i][j] * cost[j]
                elif index < 20:
                    for j in range(15):
                        column_sums[j] += seed[i][j] * output[j + 15]
                        cost_sums[j] += seed[i][j] * cost[j + 15]
                elif index < 26:
                    for j in range(15):
                        column_sums[j] += seed[i][j] * output[j + 30]
                        cost_sums[j] += seed[i][j] * cost[j + 30]
            for j in range(15):
                if column_sums[j] > count[j]:
                    fitness += count[j] * price[j] - cost_sums[j] + (column_sums[j] - count[j]) * price[j] / 2
                else:
                    fitness += column_sums[j] * price[j] - cost_sums[j]
        return fitness

    # 输出种群信息
    def info(self, array):
        self.get_chrom(array)
        profit = []
        for i in range(7):
            fitness = 0
            column_sums = [0] * 15
            cost_sums = [0] * 15
            for index, seed in enumerate(array):
                if index < 6:
                    for j in range(15):
                        column_sums[j] += seed[i][j] * output[j]
                        cost_sums[j] += seed[i][j] * cost[j]
                elif index < 20:
                    for j in range(15):
                        column_sums[j] += seed[i][j] * output[j + 15]
                        cost_sums[j] += seed[i][j] * cost[j + 15]
                elif index < 26:
                    for j in range(15):
                        column_sums[j] += seed[i][j] * output[j + 30]
                        cost_sums[j] += seed[i][j] * cost[j + 30]
            for j in range(15):
                if column_sums[j] > count[j]:
                    fitness += count[j] * price[j] - cost_sums[j] + (column_sums[j] - count[j]) * price[j] / 2
                else:
                    fitness += column_sums[j] * price[j] - cost_sums[j]
            profit.append(fitness)
        return profit

    def cross_sub(self):
        if self.select_num % 2 == 0:
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num + 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i], self.sub_sel[i + 1] = self.cross_func(self.sub_sel[i], self.sub_sel[i + 1])

    def cross_func(self, array1, array2):
        origin_1 = copy.deepcopy(array1)
        origin_2 = copy.deepcopy(array2)
        seed = random.choice(range(26))
        pos1 = random.choice(range(7))
        pos2 = random.choice(range(7))

        # 交换元素
        temp = array1[seed][pos1]
        array1[seed][pos1] = array2[seed][pos2]
        array2[seed][pos2] = temp

        # 修正违反约束条件的情况
        def is_valid(seed):
            for i in range(15):
                for j in range(6):
                    if seed[j][i] != 0 and seed[j + 1][i] != 0:
                        return False
            return True

        if not (is_valid(array1[seed]) and is_valid(array2[seed])):
            array1 = origin_1
            array2 = origin_2
        if not (is_valid(array1[seed]) and is_valid(array2[seed])):
            print("error")

        return array1, array2

    def mutation_sub(self):
        for i in range(int(self.select_num)):  # 遍历每一个 选择的子代
            if np.random.rand() <= self.pmuta_prob:  # 如果随机数小于变异概率
                seed = random.choice(range(26))
                pos1 = random.choice(range(7))
                pos2 = random.choice(range(7))
                while pos2 == pos1:  # 如果相同
                    pos2 = random.choice(range(7))
                origin = copy.deepcopy(self.sub_sel[i][seed])
                temp = self.sub_sel[i][seed][pos1]
                self.sub_sel[i][seed][pos1] = self.sub_sel[i][seed][pos2]
                self.sub_sel[i][seed][pos2] = temp

                def is_valid(seed):
                    for i in range(15):
                        for j in range(6):
                            if seed[j][i] != 0 and seed[j + 1][i] != 0:
                                return False
                    return True

                if not is_valid(self.sub_sel[i][seed]):
                    self.sub_sel[i][seed] = origin
                if not is_valid(self.sub_sel[i][seed]):
                    print("error")
        for i in range(int(self.select_num)):  # 遍历每一个选择的子代
            if np.random.rand() <= self.pmuta_prob:  # 如果随机数小于变异概率
                seed = random.choice(range(26))
                pos1 = random.choice(range(7))
                origin = copy.deepcopy(self.sub_sel[i][seed])
                self.sub_sel[i][seed][pos1] = self.get_rand_ch(seed)

                def is_valid(seed):
                    for i in range(15):
                        for j in range(6):
                            if seed[j][i] != 0 and seed[j + 1][i] != 0:
                                return False
                    return True

                if not is_valid(self.sub_sel[i][seed]):
                    self.sub_sel[i][seed] = origin
                if not is_valid(self.sub_sel[i][seed]):
                    print("error")

    def select_sub(self):
        fit = self.fitness
        sum_fit = np.cumsum(fit)
        pick = sum_fit[-1] / self.select_num * (np.random.rand() + np.array(range(int(self.select_num))))
        i, j = 0, 0
        index = []

        # 按照比例选择个体
        while i < self.size_pop and j < self.select_num:
            if sum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1

        # 深拷贝选中的个体
        self.sub_sel = [copy.deepcopy(self.chrom[x]) for x in index]

    def reins(self):
        index = np.argsort(self.fitness)[::1]
        for i in range(self.select_num):
            self.chrom[index[i]] = self.sub_sel[i]

    def check(self):
        for sub_chrom in self.chrom:
            for seed in sub_chrom:
                for i in range(6):
                    for j in range(15):
                        if seed[i][j] != 0 and seed[i + 1][j] != 0:
                            print("error")
                            print(seed)

if __name__ == "__main__":
    df = pd.read_excel('./附件1.xlsx', sheet_name=1)
    df.drop([41, 42, 43, 44], inplace=True)
    count = df['预测销量'].values.tolist()
    count2 = df['预测销量2'].values.tolist()
    df = pd.read_excel('./附件1.xlsx', sheet_name=0)
    S = df['地块面积/亩'].tolist()
    for i in range(26, 54):
        S.append(S[i])
    df = pd.read_excel('./handled_data.xlsx')
    price = df['单价'].tolist()
    output = df['亩产量/斤'].tolist()
    cost = df['种植成本/(元/亩)'].tolist()
    df = pd.read_excel('./附件2.xlsx', sheet_name=0)
    prev = df['作物编号'].tolist()
    module = GA(count, count2, S, price, output, cost, prev)
    module.rand_chrom()
    module.check()
    print("check over")
    for i in range(module.maxgen):
        module.select_sub()
        module.cross_sub()
        module.mutation_sub()
        module.reins()

        for j in range(module.size_pop):
            module.fitness[j] = module.comp_fit(module.chrom[j])

        index = module.fitness.argmax()
        if (i + 1) % 100 == 0:
            print('第' + str(i + 1) + '代后的最大适应度：' + str(module.fitness[index]))
            print('第' + str(i + 1) + '代后的最优染色体：')
            print(module.chrom[index])
            profit = module.info(module.chrom[index])
            print(profit)

        # 存储每一步的最优路径及距离
        module.best_fit.append(module.fitness[index])
        module.best_path.append(module.chrom[index])
