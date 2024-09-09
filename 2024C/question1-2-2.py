# 计算第一问第二种情况下第二部分子染色体2的遗传算法

import pandas as pd
import numpy as np
from math import floor
import time
import matplotlib.pyplot as plt
import random
import copy

class GA(object):
    def __init__(self, count, count2, S, price, output, cost,
                 maxgen=2000,
                 size_pop=1000,
                 cross_prob=0.80,
                 pmuta_prob=0.02,
                 select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.count = count
        self.count2 = count2
        self.S = S
        self.price = price
        self.output = output
        self.cost = cost

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        self.chrom = []
        self.sub_sel = []
        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []
        self.best_path = []

        self.prev = [[20], [28], [21], [22], [17], [18], [16], [16], [18], [24], [25], [26], [28], [27], [19], [19],
                     [18], [17], [17], [22],
                     [21], [29], [30, 27], [31], [32, 33], [25, 26], [17], [19], [36], [35], [35], [35], [36], [37],
                     [0], [0], [38], [38], [38], [39], [39], [39],
                     [40], [40], [40], [41], [41], [41], [41], [41], [41], [41], [24, 21], [22, 29], [28, 30], [34, 23]]
        self.prev = [[element - 1 for element in sublist] for sublist in self.prev]
    def rand_chrom(self):
        for i in range(self.size_pop):
            rand_ch = [np.zeros((7, 26)) for _ in range(56)]
            prev_used_indices = set()
            for index, seed in enumerate(rand_ch):
                flag = 0
                for pos, row in enumerate(seed):
                    if index < 8:
                        if pos == 0 and (index == 7 or index == 8):
                            flag = 1
                        num_to_generate = np.random.choice([1, 2])

                        max_sum = self.S[index + 26]
                        select_max = max_sum / 3 * 2
                        min_sum = max_sum / 3

                        if num_to_generate == 1:
                            if flag:
                                available_indices = list(set(range(1, 19)))
                                flag = 0
                                idx1 = np.random.choice(available_indices)
                                row[idx1] = max_sum

                            else:
                                available_indices = list(set(range(19)))
                                idx1 = np.random.choice(available_indices)
                                row[idx1] = max_sum
                                if idx1 == 0:
                                    flag = 1

                        else:
                            if flag:
                                flag = 0
                            available_indices = list(set(range(1, 19)))
                            idx1, idx2 = np.random.choice(available_indices, 2, replace=False)

                            value1 = round(np.random.uniform(min_sum, select_max), 2)
                            value2 = max_sum - value1

                            row[idx1] = value1
                            row[idx2] = value2


                    elif index < 24:
                        num_to_generate = np.random.choice([1, 2])

                        max_sum = self.S[index + 26]
                        select_max = max_sum / 3 * 2
                        min_sum = max_sum / 3

                        if num_to_generate == 1:
                            available_indices = list(set(range(1, 19)))
                            idx1 = np.random.choice(available_indices)
                            row[idx1] = max_sum
                        else:
                            available_indices = list(set(range(1, 19)))
                            idx1, idx2 = np.random.choice(available_indices, 2, replace=False)

                            value1 = round(np.random.uniform(min_sum, select_max), 2)
                            value2 = max_sum - value1

                            row[idx1] = value1
                            row[idx2] = value2

                    elif index < 28:
                        if pos == 0:
                            prev_used_indices = set(self.prev[index])
                        else:
                            non_zero_indices_cur_row = np.nonzero(rand_ch[index + 28][pos - 1])[0]
                            prev_used_indices = set(non_zero_indices_cur_row)

                        num_to_generate = np.random.choice([1, 2])
                        max_sum = 0.6
                        select_max = max_sum / 3 * 2
                        min_sum = max_sum / 3

                        if num_to_generate == 1:
                            available_indices = list(set(range(1, 19)) - prev_used_indices)
                            idx1 = np.random.choice(available_indices)
                            row[idx1] = max_sum
                            prev_used_indices = {idx1}
                        else:
                            available_indices = list(set(range(1, 19)) - prev_used_indices)
                            idx1, idx2 = np.random.choice(available_indices, 2, replace=False)

                            value1 = round(np.random.uniform(min_sum, select_max), 2)
                            value2 = max_sum - value1

                            row[idx1] = value1
                            row[idx2] = value2
                            prev_used_indices = {idx2}

                        num_to_generate = np.random.choice([1, 2])
                        max_sum = 0.6
                        select_max = max_sum / 3 * 2
                        min_sum = max_sum / 3

                        if num_to_generate == 1:
                            available_indices = list(set(range(1, 19)) - prev_used_indices)
                            idx1 = np.random.choice(available_indices)
                            rand_ch[index + 28][pos][idx1] = max_sum
                        else:
                            available_indices = list(set(range(1, 19)) - prev_used_indices)
                            idx1, idx2 = np.random.choice(available_indices, 2, replace=False)

                            value1 = round(np.random.uniform(min_sum, select_max), 2)
                            value2 = max_sum - value1

                            rand_ch[index + 28][pos][idx1] = value1
                            rand_ch[index + 28][pos][idx2] = value2

                    elif index < 36:
                        if rand_ch[index - 28][pos][0] != 0:
                            continue
                        num_to_generate = np.random.choice([1, 2])

                        max_sum = self.S[index + 26]
                        select_max = max_sum / 3 * 2
                        min_sum = max_sum / 3

                        if num_to_generate == 1:
                            available_indices = list(set(range(19, 22)))
                            idx1 = np.random.choice(available_indices)
                            row[idx1] = max_sum
                        else:
                            available_indices = list(set(range(19, 22)))
                            idx1, idx2 = np.random.choice(available_indices, 2, replace=False)

                            value1 = round(np.random.uniform(min_sum, select_max), 2)
                            value2 = max_sum - value1

                            row[idx1] = value1
                            row[idx2] = value2


                    elif index >= 36 and index < 52:
                        num_to_generate = np.random.choice([1, 2])

                        max_sum = 0.6
                        select_max = max_sum / 3 * 2
                        min_sum = max_sum / 3

                        if num_to_generate == 1:
                            available_indices = list(set(range(22, 26)))
                            idx1 = np.random.choice(available_indices)
                            row[idx1] = max_sum
                        else:
                            available_indices = list(set(range(22, 26)))
                            idx1, idx2 = np.random.choice(available_indices, 2, replace=False)

                            value1 = round(np.random.uniform(min_sum, select_max), 2)
                            value2 = max_sum - value1

                            row[idx1] = value1
                            row[idx2] = value2
                    elif index >= 52:
                        break

            self.get_chrom(rand_ch)
            # 生成染色体并计算适应度
            self.fitness[i] = self.comp_fit(rand_ch)
            self.chrom.append(rand_ch)

    def get_chrom(self, rand_ch):
        for index, seed in enumerate(rand_ch):
            try:
                if index < 24:
                    for j in range(0, 5, 1):
                        sum_first_five_cols = np.sum(seed[j:j + 3, 1:4])

                        while sum_first_five_cols < self.S[index + 26]:
                            non_zero_indices = np.argwhere(seed[j:j + 3, 4:] != 0)

                            row_to_modify, col_to_modify = 0, 0
                            if len(non_zero_indices) != 0:
                                row_to_modify, col_to_modify = non_zero_indices[np.random.choice(len(non_zero_indices))]
                                row_to_modify += j
                                col_to_modify += 4
                            else:
                                row_to_modify = j + 2
                                col_to_modify = 0

                            non_zero_indices_cur_row = np.nonzero(seed[row_to_modify][1:4])[0] + 1
                            available_positions = list(set(range(1,4)) - set(non_zero_indices_cur_row))
                            new_col_index = np.random.choice(available_positions)

                            seed[row_to_modify, new_col_index] = seed[row_to_modify, col_to_modify]
                            seed[row_to_modify, col_to_modify] = 0

                            # Recalculate the sum after modification
                            sum_first_five_cols = np.sum(seed[j:j + 3, 1:4])

                elif index < 28:
                    for j in range(0, 5, 1):
                        sum_first_five_cols = np.sum(seed[j:j + 3, 1:4])
                        sum_first_five_cols2 = np.sum(rand_ch[index+28][j:j + 3, 1:4])
                        sum = sum_first_five_cols + sum_first_five_cols2
                        while sum < self.S[index+26]:
                            choice = np.random.choice([0, 1])
                            available_positions = []
                            row_to_modify, col_to_modify = 0, 0
                            if choice:
                                while len(available_positions) == 0:
                                    non_zero_indices = np.argwhere(seed[j:j + 3, 4:] != 0)

                                    row_to_modify, col_to_modify = non_zero_indices[np.random.choice(len(non_zero_indices))]

                                    row_to_modify += j
                                    col_to_modify += 4

                                    non_zero_indices_prev_row = np.nonzero(rand_ch[index+28][row_to_modify - 1][1:4])[
                                        0] + 1 if row_to_modify - 1 >= 0 else set()
                                    non_zero_indices_cur_row = np.nonzero(seed[row_to_modify][1:4])[0] + 1
                                    non_zero_indices_next_row = np.nonzero(rand_ch[index+28][row_to_modify][1:4])[0] + 1
                                    available_positions = list(
                                        set(range(1, 4)) - set(non_zero_indices_prev_row) - set(
                                            non_zero_indices_cur_row) - set(
                                            non_zero_indices_next_row))

                                new_col_index = np.random.choice(available_positions)

                                seed[row_to_modify, new_col_index] = seed[row_to_modify, col_to_modify]
                                seed[row_to_modify, col_to_modify] = 0
                            else:
                                while len(available_positions) == 0:
                                    non_zero_indices = np.argwhere(rand_ch[index+28][j:j + 3, 4:] != 0)

                                    row_to_modify, col_to_modify = non_zero_indices[np.random.choice(len(non_zero_indices))]

                                    row_to_modify += j
                                    col_to_modify += 4

                                    non_zero_indices_prev_row = np.nonzero(seed[row_to_modify][1:4])[
                                        0] + 1 if row_to_modify - 1 >= 0 else set()
                                    non_zero_indices_cur_row = np.nonzero(rand_ch[index+28][row_to_modify][1:4])[0] + 1
                                    non_zero_indices_next_row = np.nonzero(seed[row_to_modify][1:4])[0] + 1
                                    if row_to_modify + 1 < 7:
                                        non_zero_indices_next_row = np.nonzero(seed[row_to_modify + 1][1: 4])[0] + 1
                                    available_positions = list(
                                        set(range(1, 4)) - set(non_zero_indices_prev_row) - set(
                                            non_zero_indices_cur_row) - set(
                                            non_zero_indices_next_row))

                                new_col_index = np.random.choice(available_positions)

                                rand_ch[index+28][row_to_modify, new_col_index] = rand_ch[index+28][row_to_modify, col_to_modify]
                                rand_ch[index+28][row_to_modify, col_to_modify] = 0

                            sum_first_five_cols = np.sum(seed[j:j + 3, 1:4])
                            sum_first_five_cols2 = np.sum(rand_ch[index + 28][j:j + 3, 1:4])
                            sum = sum_first_five_cols + sum_first_five_cols2
                else:
                    break
            except ValueError as e:
                print(f"Error occurred with seed at index {index}: {e}")
                print("Current seed state:")
                print(seed)
                raise

        return rand_ch

    def info(self, array):
        lfitness_1 = []
        lfitness_2 = []
        for i in range(7):
            fitness_1 = 0
            column_sums = [0] * 26
            cost_sums = [0] * 26
            for index, seed in enumerate(array):
                if index < 8:
                    for j in range(19):
                        column_sums[j] += seed[i][j] * output[j]
                        cost_sums[j] += seed[i][j] * cost[j]
                elif index < 24:
                    for j in range(1, 19):
                        column_sums[j] += seed[i][j] * output[j + 19 - 1]
                        cost_sums[j] += seed[i][j] * cost[j + 19 - 1]
                elif index < 28:
                    for j in range(1, 19):
                        column_sums[j] += seed[i][j] * output[j + 44 - 1]
                        cost_sums[j] += seed[i][j] * cost[j + 44 - 1]
            for j in range(19):
                if column_sums[j] > count[j + 15]:
                    fitness_1 += count[j + 15] * price[j] - cost_sums[j] + (column_sums[j] - count[j + 15]) * price[j] / 2
                else:
                    fitness_1 += column_sums[j] * price[j] - cost_sums[j]
            lfitness_1.append(fitness_1)
        for i in range(7):
            fitness_2 = 0
            column_sums = [0] * 26
            cost_sums = [0] * 26
            for index, seed in enumerate(array):
                if index < 28:
                    continue
                elif index < 52:
                    for j in range(19, 26):
                        column_sums[j] += seed[i][j] * output[j + 37 - 19]
                        cost_sums[j] += seed[i][j] * cost[j + 37 - 19]
                else:
                    for j in range(1, 19):
                        column_sums[j] += seed[i][j] * output[j + 44 - 1]
                        cost_sums[j] += seed[i][j] * cost[j + 44 - 1]

            for j in range(1, 26):
                if j < 19:
                    if column_sums[j] > count2[j + 16 - 1]:
                        fitness_2 += count2[j + 16 - 1] * price[j + 44 - 1] - cost_sums[j] + (column_sums[j] - count2[j + 16 - 1]) * price[j + 44 - 1] / 2
                    else:
                        fitness_2 += column_sums[j] * price[j + 44 - 1] - cost_sums[j]
                else:
                    if column_sums[j] > count2[j + 16 - 1]:
                        fitness_2 += count2[j + 16 - 1] * price[j + 37 - 1] - cost_sums[j] + (column_sums[j] - count2[j + 16 - 1]) * price[j + 37 - 1] / 2
                    else:
                        fitness_2 += column_sums[j] * price[j + 37 - 1] - cost_sums[j]
            lfitness_2.append(fitness_2)
        return lfitness_1, lfitness_2

    def comp_fit(self, array):
        self.get_chrom(array)
        fitness_1 = 0
        fitness_2 = 0
        for i in range(7):
            column_sums = [0] * 26
            cost_sums = [0] * 26
            for index, seed in enumerate(array):
                non_zero_indices_cur_row = np.nonzero(seed[0])[0]
                non_zero_set = set(non_zero_indices_cur_row)
                if any(elem in non_zero_set for elem in self.prev[index]):
                    return 0
                if index < 8:
                    for j in range(19):
                        column_sums[j] += seed[i][j] * output[j]
                        cost_sums[j] += seed[i][j] * cost[j]
                elif index < 24:
                    for j in range(1, 19):
                        column_sums[j] += seed[i][j] * output[j + 19 - 1]
                        cost_sums[j] += seed[i][j] * cost[j + 19 - 1]
                elif index < 28:
                    for j in range(1, 19):
                        column_sums[j] += seed[i][j] * output[j + 44 - 1]
                        cost_sums[j] += seed[i][j] * cost[j + 44 - 1]
            for j in range(19):
                if column_sums[j] > count[j + 15]:
                    fitness_1 += count[j + 15] * price[j] - cost_sums[j] + (column_sums[j] - count[j + 15]) * price[j] / 2
                else:
                    fitness_1 += column_sums[j] * price[j] - cost_sums[j]
        for i in range(7):
            column_sums = [0] * 26
            cost_sums = [0] * 26
            for index, seed in enumerate(array):
                if index < 28:
                    continue
                elif index < 52:
                    for j in range(19, 26):
                        column_sums[j] += seed[i][j] * output[j + 37 - 19]
                        cost_sums[j] += seed[i][j] * cost[j + 37 - 19]
                else:
                    for j in range(1, 19):
                        column_sums[j] += seed[i][j] * output[j + 44 - 1]
                        cost_sums[j] += seed[i][j] * cost[j + 44 - 1]

            for j in range(1, 26):
                if j < 19:
                    if column_sums[j] > count2[j + 16 - 1]:
                        fitness_2 += count2[j + 16 - 1] * price[j + 44 - 1] - cost_sums[j] + (column_sums[j] - count2[j + 16 - 1]) * price[j + 44 - 1] / 2
                    else:
                        fitness_2 += column_sums[j] * price[j + 44 - 1] - cost_sums[j]
                else:
                    if column_sums[j] > count2[j + 16 - 1]:
                        fitness_2 += count2[j + 16 - 1] * price[j + 37 - 1] - cost_sums[j] + (column_sums[j] - count2[j + 16 - 1]) * price[j + 37 - 1] / 2
                    else:
                        fitness_2 += column_sums[j] * price[j + 37 - 1] - cost_sums[j]
        fitness = fitness_1 + fitness_2
        return fitness

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
        seed = random.choice(range(56))
        pos1 = random.choice(range(7))
        pos2 = random.choice(range(7))

        while array1[seed][pos1][0] != 0 or np.count_nonzero(array1[seed][pos1]) == 0:
            pos1 = random.choice(range(7))
        while array2[seed][pos2][0] != 0 or np.count_nonzero(array1[seed][pos1]) == 0:
            pos2 = random.choice(range(7))
        if seed < 24 or (seed >= 28 and seed < 52):
            temp = array1[seed][pos1]
            array1[seed][pos1] = array2[seed][pos2]
            array2[seed][pos2] = temp


        return array1, array2

    def mutation_sub(self):
        for i in range(int(self.select_num)):  # 遍历每一个 选择的子代
            if np.random.rand() <= self.pmuta_prob:  # 如果随机数小于变异概率
                seed = random.choice(range(56))
                pos1 = random.choice(range(7))
                pos2 = random.choice(range(7))
                while pos2 == pos1:
                    pos2 = random.choice(range(7))
                while self.sub_sel[i][seed][pos1][0] != 0 or np.count_nonzero(self.sub_sel[i][seed][pos1]) == 0:
                    pos1 = random.choice(range(7))
                while self.sub_sel[i][seed][pos2][0] != 0 or np.count_nonzero(self.sub_sel[i][seed][pos1]) == 0:
                    pos2 = random.choice(range(7))
                origin = copy.deepcopy(self.sub_sel[i][seed])
                if seed < 24 or (seed >= 28 and seed < 52):
                    temp = self.sub_sel[i][seed][pos1]
                    self.sub_sel[i][seed][pos1] = self.sub_sel[i][seed][pos2]
                    self.sub_sel[i][seed][pos2] = temp
                elif seed < 28:
                    prev_used_indices = set()
                    prev_used_indices2 = set()
                    if pos1 == 0:
                        prev_used_indices = set(self.prev[seed])
                        non_zero_indices_cur_row = np.nonzero(self.sub_sel[i][seed + 28][pos1])[0]
                        prev_used_indices2 = set(non_zero_indices_cur_row)
                    else:
                        non_zero_indices_cur_row = np.nonzero(self.sub_sel[i][seed + 28][pos1 - 1])[0]
                        prev_used_indices = set(non_zero_indices_cur_row)
                        non_zero_indices_cur_row = np.nonzero(self.sub_sel[i][seed + 28][pos1])[0]
                        prev_used_indices2 = set(non_zero_indices_cur_row)
                    num_to_generate = np.random.choice([1, 2])
                    max_sum = 0.6
                    select_max = max_sum / 3 * 2
                    min_sum = max_sum / 3
                    rand = np.zeros(26)
                    if num_to_generate == 1:
                        available_indices = list(set(range(1, 19)) - prev_used_indices - prev_used_indices2)
                        idx1 = np.random.choice(available_indices)
                        rand[idx1] = max_sum
                    else:
                        available_indices = list(set(range(1, 19)) - prev_used_indices - prev_used_indices2)
                        idx1, idx2 = np.random.choice(available_indices, 2, replace=False)

                        value1 = round(np.random.uniform(min_sum, select_max), 2)
                        value2 = max_sum - value1

                        rand[idx1] = value1
                        rand[idx2] = value2
                    self.sub_sel[i][seed][pos1] = rand
                else:
                    prev_used_indices = set()
                    prev_used_indices2 = set()
                    if pos1 == 6:
                        non_zero_indices_cur_row = np.nonzero(self.sub_sel[i][seed - 28][pos1])[0]
                        prev_used_indices = set(non_zero_indices_cur_row)
                    else:
                        non_zero_indices_cur_row = np.nonzero(self.sub_sel[i][seed - 28][pos1])[0]
                        prev_used_indices = set(non_zero_indices_cur_row)
                        non_zero_indices_cur_row = np.nonzero(self.sub_sel[i][seed - 28][pos1 + 1])[0]
                        prev_used_indices2 = set(non_zero_indices_cur_row)
                    num_to_generate = np.random.choice([1, 2])
                    max_sum = 0.6
                    select_max = max_sum / 3 * 2
                    min_sum = max_sum / 3
                    rand = np.zeros(26)
                    if num_to_generate == 1:
                        available_indices = list(set(range(1, 19)) - prev_used_indices - prev_used_indices2)
                        idx1 = np.random.choice(available_indices)
                        rand[idx1] = max_sum
                    else:
                        available_indices = list(set(range(1, 19)) - prev_used_indices - prev_used_indices2)
                        idx1, idx2 = np.random.choice(available_indices, 2, replace=False)

                        value1 = round(np.random.uniform(min_sum, select_max), 2)
                        value2 = max_sum - value1

                        rand[idx1] = value1
                        rand[idx2] = value2
                    self.sub_sel[i][seed][pos1] = rand


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


if __name__ == "__main__":
    seed = 60
    random.seed(seed)
    np.random.seed(seed)
    df = pd.read_excel('./附件1.xlsx', sheet_name=1)
    df.drop([41, 42, 43, 44], inplace=True)
    count = df['预测销量'].values.tolist()
    count2 = df['预测销量2'].values.tolist()
    df = pd.read_excel('./附件1.xlsx', sheet_name=0)
    S = df['地块面积/亩'].tolist()
    for i in range(26, 54):
        S.append(S[i])
    df = pd.read_excel('./handled_data.xlsx')
    price = df['单价'].tolist()[45:]
    output = df['亩产量/斤'].tolist()[45:]
    cost = df['种植成本/(元/亩)'].tolist()[45:]
    module = GA(count, count2, S, price, output, cost)
    module.rand_chrom()
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
            if (i + 1) % 500 == 0:
                print(module.chrom[index])
            l1, l2 = module.info(module.chrom[index])
            print(l1, l2)

        module.best_fit.append(module.fitness[index])
        module.best_path.append(module.chrom[index])

