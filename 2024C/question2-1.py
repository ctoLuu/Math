# 计算第二问第一部分子染色体1的遗传算法和蒙塔卡洛组合优化

import pandas as pd
import numpy as np
from math import floor
import time
import matplotlib.pyplot as plt
import random
import copy

num = 100

def getCount(count):
    origin_count = copy.deepcopy(count)
    Count = [[[] for _ in range(7)] for _ in range(num)]

    corn_growth_rates = np.random.lognormal(mean=np.log(0.075), sigma=0.1, size=num * 7)
    wheat_growth_rates = np.random.lognormal(mean=np.log(0.075), sigma=0.1, size=num * 7)
    corn_growth_rates = np.clip(corn_growth_rates, 0.05, 0.10)
    wheat_growth_rates = np.clip(wheat_growth_rates, 0.05, 0.10)

    sales_growth_rates = np.random.normal(0, 0.025, num * 7 * 13)
    sales_growth_rates = np.clip(sales_growth_rates, -0.05, 0.05)

    flag_corn = 0
    flag_wheat = 0
    flag = 0

    for sub_count in Count:
        count = copy.deepcopy(origin_count)
        for i in range(7):
            for j in range(15):
                if j == 5:
                    sub_count[i].append(count[j] * (1 + wheat_growth_rates[flag_wheat]))
                    flag_wheat += 1
                elif j == 6:
                    sub_count[i].append(count[j] * (1 + corn_growth_rates[flag_corn]))
                    flag_corn += 1
                else:
                    sub_count[i].append(origin_count[j] * (1 + sales_growth_rates[flag]))
                    flag += 1
            count = copy.deepcopy(sub_count[i])
    return Count

def getOutput(output):
    # 转换为NumPy数组以便于计算
    origin_output = np.array(output)

    # 生成增长率
    output_growth_rates = np.random.normal(0, 0.05, (num, 7, 45))
    sales_growth_rates = np.clip(output_growth_rates, -0.1, 0.1)

    # 创建结果数组，避免使用嵌套列表
    Output = np.zeros((num, 7, 45))

    # 对每个 'sub_output' 进行计算
    for idx in range(num):
        # 重置 'output' 为原始的输出
        output = origin_output.copy()
        for i in range(7):
            # 计算每一列的增长后的值
            Output[idx, i] = output * (1 + sales_growth_rates[idx, i])
            # 更新 'output' 为最新一轮的结果，用于下一轮
            output = Output[idx, i]

    return Output

def getCost(cost):
    # 转换为 NumPy 数组以便于计算
    origin_cost = np.array(cost)

    # 固定的年增长率5%
    growth_rate = 1.05

    # 创建结果数组，避免使用嵌套列表
    Costs = np.zeros((num, 7, 45))

    # 对每年的成本进行计算
    for idx in range(num):
        # 重置 'cost' 为原始的种植成本
        current_cost = origin_cost.copy()
        for i in range(7):
            # 计算每天的成本变化，成本每年增加5%
            Costs[idx, i] = current_cost * growth_rate
            # 更新 'current_cost' 为最新一轮的结果，用于下一轮
            current_cost = Costs[idx, i]

    return Costs

class GA(object):
    def __init__(self, count, count2, S, price, output, cost, prev,
                 maxgen=2000,
                 size_pop=100,
                 cross_prob=0.80,
                 pmuta_prob=0.02,
                 select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.num = 26 * 7
        self.count = getCount(count[:45])
        self.count2 = count2
        self.S = S
        self.price = price
        self.output = getOutput(output[:45])
        self.cost = getCost(cost[:45])

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        self.chrom = []
        self.sub_sel = []
        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []
        self.best_path = []
        self.prev = prev

    def get_rand_ch(self, index):
        seed = np.zeros(15)
        available_indices = list(set(range(15)))
        num_to_generate = np.random.choice([1, 2])

        max_sum = self.S[index]
        select_max = max_sum / 3 * 2
        min_sum = int(max_sum / 3)

        if num_to_generate == 1:
            idx1 = np.random.choice(available_indices)
            seed[idx1] = max_sum
        else:
            idx1, idx2 = np.random.choice(available_indices, 2, replace=False)

            value1 = np.random.randint(min_sum, select_max + 1)
            value2 = max_sum - value1

            seed[idx1] = value1
            seed[idx2] = value2
        return seed
    def rand_chrom(self):
        for i in range(self.size_pop):
            rand_ch = [np.zeros((7, 15)) for _ in range(26)]
            for index, seed in enumerate(rand_ch):
                prev_used_indices = set()
                if index == 0:
                    prev_used_indices = set([self.prev[index] - 1])
                for row in seed:
                    available_indices = list(set(range(15)) - prev_used_indices)
                    num_to_generate = np.random.choice([1, 2])

                    max_sum = self.S[index]
                    select_max = max_sum / 3 * 2
                    min_sum = int(max_sum / 3)

                    if num_to_generate == 1:
                        idx1 = np.random.choice(available_indices)
                        row[idx1] = max_sum

                        prev_used_indices = {idx1}
                    else:
                        if len(available_indices) >= 2:
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
            self.get_chrom(rand_ch)
            # 生成染色体并计算适应度
            self.fitness[i], _ = self.comp_fit(rand_ch)
            self.chrom.append(rand_ch)
        self.check()

    def get_chrom(self, rand_ch):
        for index, seed in enumerate(rand_ch):
            try:
                for j in range(0, 5, 1):
                    sum_first_five_cols = np.sum(seed[j:j + 3, :5])

                    while sum_first_five_cols < self.S[index]:
                        non_zero_indices = np.argwhere(seed[j:j + 3, 5:] != 0)

                        if len(non_zero_indices) == 0:
                            raise ValueError("No non-zero elements found to modify")

                        row_to_modify, col_to_modify = non_zero_indices[np.random.choice(len(non_zero_indices))]

                        row_to_modify += j
                        col_to_modify += 5

                        non_zero_indices_prev_row = np.nonzero(seed[row_to_modify - 1][:5])[
                            0] if row_to_modify - 1 >= 0 else set()
                        non_zero_indices_cur_row = np.nonzero(seed[row_to_modify][:5])[0]
                        non_zero_indices_next_row = np.nonzero(seed[row_to_modify][:5])[0]
                        if row_to_modify + 1 < 7:
                            non_zero_indices_next_row = np.nonzero(seed[row_to_modify + 1][:5])[0]
                        available_positions = list(
                            set(range(5)) - set(non_zero_indices_prev_row) - set(non_zero_indices_cur_row) - set(non_zero_indices_next_row))

                        if not available_positions:
                            raise ValueError("No available positions to move the value to")

                        new_col_index = np.random.choice(available_positions)

                        if row_to_modify + 1 < len(seed):
                            non_zero_indices_next_row = np.nonzero(seed[row_to_modify + 1][:5])[0]
                            if new_col_index in non_zero_indices_next_row:
                                available_positions.remove(new_col_index)
                                if not available_positions:
                                    raise ValueError("No available positions to move the value to")
                                new_col_index = np.random.choice(available_positions)

                        seed[row_to_modify, new_col_index] = seed[row_to_modify, col_to_modify]
                        seed[row_to_modify, col_to_modify] = 0

                        sum_first_five_cols = np.sum(seed[j:j + 3, :5])

            except ValueError as e:
                # 捕获到错误时打印 seed 和错误信息
                print(f"Error occurred with seed at index {index}: {e}")
                print("Current seed state:")
                print(seed)
                raise  # 重新引发异常，以便可以在其他地方处理或停止执行

        return rand_ch


    def comp_fit(self, array):
        self.get_chrom(array)
        fitness_ = 0
        fitness_list = []
        fitness_year = []
        for epoch in range(num):
            fitness = 0
            years = []
            for i in range(7):
                year_fitness = 0
                column_sums = [0] * 15
                cost_sums = [0] * 15
                for index, seed in enumerate(array):
                    non_zero_indices_cur_row = np.nonzero(seed[0])[0]
                    if self.prev[index] - 1 in non_zero_indices_cur_row:
                        return 0, None
                    if index < 6:
                        for j in range(15):
                            column_sums[j] += seed[i][j] * self.output[epoch][i][j]
                            cost_sums[j] += seed[i][j] * self.cost[epoch][i][j]
                    elif index < 20:
                        for j in range(15):
                            column_sums[j] += seed[i][j] * self.output[epoch][i][j + 15]
                            cost_sums[j] += seed[i][j] * self.cost[epoch][i][j + 15]
                    elif index < 26:
                        for j in range(15):
                            column_sums[j] += seed[i][j] * self.output[epoch][i][j + 30]
                            cost_sums[j] += seed[i][j] * self.cost[epoch][i][j + 30]
                for j in range(15):
                    if column_sums[j] > self.count[epoch][i][j]:
                        year_fitness += self.count[epoch][i][j] * price[j] - cost_sums[j] + (column_sums[j] - count[j]) * price[j] / 2
                    else:
                        year_fitness += column_sums[j] * price[j] - cost_sums[j]
                years.append(year_fitness)
                fitness += year_fitness
            fitness_list.append(fitness)
            fitness_year.append(years)
        fitness_ = np.mean(fitness_list)
        mean_list = [sum(x) / len(x) for x in zip(*fitness_year)]
        return fitness_, mean_list

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
        for i in range(int(self.select_num)):  # 遍历每一个 选择的子代
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
            module.fitness[j], _ = module.comp_fit(module.chrom[j])

        index = module.fitness.argmax()
        if (i + 1) % 10 == 0:
            print('第' + str(i + 1) + '代后的最大适应度：' + str(module.fitness[index]))
            print('第' + str(i + 1) + '代后的最优染色体：')
            print(module.chrom[index])
            _, year_list = module.comp_fit(module.chrom[index])
            print(year_list)

        # 存储每一步的最优路径及距离
        module.best_fit.append(module.fitness[index])
        module.best_path.append(module.chrom[index])


