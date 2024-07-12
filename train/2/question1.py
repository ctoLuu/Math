import pandas as pd
import numpy as np
import time
from math import floor
import random
from copy import deepcopy


class GA(object):
    def __init__(self, time_matrix, send_array, maxgen=400, size_pop=5000, cross_prob=0.80, pmuta_prob=0.02, select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.punishment = 99999999999
        self.num = 43
        self.time_matrix = time_matrix
        self.send_array = send_array

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        # self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop, self.num)
        # self.chrom = np.array([np.array([0] * 43) for _ in range(self.size_pop)])
        # self.sub_sel = np.array([0] * int(self.select_num) * self.num).reshape(self.select_num, self.num)
        # self.sub_sel = np.array([np.array([0] * 43) for _ in range(int(self.select_num))])
        self.chrom = []
        self.sub_sel = []

        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []
        self.best_path = []

    def rand_chrom(self):
        rand_ch = np.array(range(1, self.num + 1))
        for i in range(self.size_pop):
            rand_ch = np.array(range(1, self.num + 1))
            np.random.shuffle(rand_ch)
            rand_ch = self.encode(rand_ch)

            self.fitness[i], rand_ch = self.comp_fit(rand_ch)
            self.chrom.append(rand_ch)

    def encode(self, array):
        encode_array = np.insert(array, 0, 0)
        current_time = 0
        # for index, value in np.ndenumerate(encode_array):
        #     if index == len(encode_array)-1:
        #         break
        #     next_time = time_matrix[value, encode_array[index+1]]
        #     current_time += next_time
        #     if current_time > 8:
        #         current_time = 0
        #         encode_array = np.insert(array, index, 0)
        index = 0
        while index != len(encode_array)-1:
            next_time = time_matrix[encode_array[index], encode_array[index+1]]
            current_time += next_time
            if current_time > 8:
                current_time = 0
                encode_array = np.insert(encode_array, index+1, 0)
            index += 1
        return encode_array

    def decode(self, array):
        decode_array = array[array > 0]
        return decode_array

    def comp_fit(self, array):
        fitness = 0
        zero_indices = np.where(array == 0)[0]
        sub_arrays = []
        start_index = 0
        for index in zero_indices:
            sub_arrays.append(array[start_index:index])
            start_index = index + 1
        sub_arrays.append(array[start_index:])

        flag = 0
        new_array = []
        for arr in sub_arrays:
            if len(arr) == 0:
                continue
            time_list = []
            for i in arr:
                time_list.append(self.send_array[i])
            if len(time_list) < 3:
                new_array += list(arr)
                fitness += 1
            else:
                for i in range(len(time_list) - 2):
                    if time_list[i] == 0 and time_list[i + 1] == 1 and time_list[i + 2] == 0:
                        res, sub_array = self.find_fit(arr, time_list)
                        new_array += list(sub_array)
                        fitness += res
                        flag = 1
                        break
                    if time_list[i] == 1 and time_list[i + 1] == 0 and time_list[i + 2] == 1:
                        res, sub_array = self.find_fit(arr, time_list)
                        new_array += list(sub_array)
                        fitness += res
                        flag = 1
                        break
                if flag == 0:
                    if time_list[time_list == 0] > 1 and time_list[0] == 1:
                        zero_indices = np.where(np.array(time_list) == 0)[0]
                        res, sub_array = self.find_fit(arr, time_list)
                        new_array += list()
                        fitness += res
                        flag = 1
                        break
                if flag == 0:
                    fitness += 1
                    new_array += list(arr)
                else:
                    flag = 0
        array = self.encode(new_array)
        return fitness, array

    def find_fit(self, array, time_list):
        res = 0
        zero_indices = np.where(np.array(time_list) == 0)[0]
        no_indices = np.where(np.array(time_list) != 0)[0]
        new_array1 = []
        time1 = 0
        time2 = 0
        for i in range(len(zero_indices)):
            new_array1.append(array[zero_indices[i]])
            time1 = self.comp_time(new_array1, 0)
        for i in range(len(no_indices)):
            new_array1.append(array[no_indices[i]])

        new_array2 = []
        for i in range(len(no_indices)):
            new_array2.append(array[no_indices[i]])

        for i in range(len(zero_indices)):
            new_array2.append(array[zero_indices[i]])

        if time1:
            encode_array1 = self.encode(new_array1)
            encode_array2 = self.encode(new_array2)
            if len(encode_array1[encode_array1 == 0]) < len(encode_array2[encode_array2 == 0]):
                return len(encode_array1[encode_array1 == 0]), self.decode(encode_array1)
            else:
                return len(encode_array2[encode_array2 == 0]), encode_array2
        else:
            encode_array1 = self.encode(new_array1)
            return len(encode_array1[encode_array1 == 0]), self.decode(encode_array1)

    def comp_time(self, array, flag):
        index = 0
        current_time = 0
        while index != len(array)-1:
            next_time = time_matrix[array[index], array[index+1]]
            current_time += next_time
            if flag == 0 and current_time > 4:
                return False
        return current_time

    def select_sub(self):
        fit = 1. / self.fitness
        sum_fit = np.cumsum(fit)
        pick = sum_fit[-1] / self.select_num * (np.random.rand() + np.array(range(int(self.select_num))))
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if sum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = [self.chrom[x] for x in index]

    def cross_sub(self):
        if self.select_num % 2 == 0:
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num + 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i], self.sub_sel[i + 1] = self.cross_func(self.sub_sel[i], self.sub_sel[i + 1])

    def cross_func(self, array1, array2):
        zero_indices1 = np.where(array1 == 0)[0]
        zero_pos1 = random.choice(range(len(zero_indices1)-1))
        zero_indices2 = np.where(array2 == 0)[0]
        zero_pos2 = random.choice(range(len(zero_indices2)-1))
        slice1 = array1[zero_indices1[zero_pos1]+1:zero_indices1[zero_pos1+1]]
        slice2 = array2[zero_indices2[zero_pos2]+1:zero_indices2[zero_pos2+1]]
        # new_array1 = np.delete(array1, slice(zero_indices1[zero_pos1], zero_indices1[zero_pos1+1]))
        # new_array2 = np.delete(array2, slice(zero_indices2[zero_pos2], zero_indices2[zero_pos2+1]))
        # new_array1 = np.concatenate((slice1, new_array1))
        # new_array2 = np.concatenate((slice2, new_array2))

        decode_slice1 = self.decode(slice1)
        decode_slice2 = self.decode(slice2)
        decode_array1 = self.decode(array1)
        decode_array2 = self.decode(array2)

        new_array1 = np.array([x for x in decode_array2 if x not in slice1])
        new_array2 = np.array([x for x in decode_array1 if x not in slice2])
        new_array1 = np.concatenate((slice1, new_array1))
        new_array2 = np.concatenate((slice2, new_array2))
        encode_array1 = self.encode(new_array1)
        encode_array2 = self.encode(new_array2)
        return encode_array1, encode_array2

    def mutation_sub(self):
        for index, array in enumerate(self.sub_sel):
            # if index < 30:
            #     continue
            if np.random.rand() <= self.pmuta_prob:
                flag, mutate_array = self.mutate_func(array)
                if flag == True:
                    self.sub_sel[index] = mutate_array

    def mutate_func(self, array):
        mutate_num = 10
        mutate_array = []
        fitness = []
        for i in range(mutate_num):
            p1, p2 = random.choices(list(i for i in range(1, len(array)) if array[i] != 0), k=2)
            new_array = array.copy()
            new_array[p1], new_array[p2] = new_array[p2], new_array[p1]
            new_array = self.decode(new_array)
            new_array = self.encode(new_array)
            fit, new_array = self.comp_fit(new_array)
            new_array = self.encode(new_array)
            fitness.append(fit)
            mutate_array.append(new_array)

        if len(fitness):
            sorted_with_index = sorted(enumerate(fitness), key=lambda x: x[1])
            min_value_index = sorted_with_index[0][0]
            return True, mutate_array[min_value_index]
        else:
            return False, []

    def reins(self):
        index = np.argsort(self.fitness)[::-1]  # 替换最差的（倒序）
        for i in range(self.select_num):
            self.chrom[index[i]] = self.sub_sel[i]

if __name__ == "__main__":
    df = pd.read_excel('./数据.xlsx', sheet_name=0)
    df.drop(columns=['配送中心', '允许到店时间段'], inplace=True)
    df['时间属性'] = df['时间属性'].map({'夜配': 0, '日配': 1})
    df2 = pd.DataFrame(['配送中心', '120.44739', '31.50353', 2]).T
    df2.columns = df.columns
    df = pd.concat([df2, df], ignore_index=True)
    df.drop(index=[3, 7, 23], inplace=True)
    df.reset_index(inplace=True)
    print(df)

    time_matrix = pd.read_excel('./time_matrix.xlsx')
    time_matrix.set_index(['到达门店简称'], inplace=True)
    time_matrix = time_matrix.to_numpy(dtype=float)
    print(time_matrix)
    send_array = df['时间属性']
    send_array = send_array.to_numpy(dtype=int)
    print(send_array)

    module = GA(time_matrix, send_array)
    module.rand_chrom()
    for i in range(module.maxgen):
        module.select_sub()
        module.cross_sub()
        module.mutation_sub()
        # module.reverse_sub()
        module.reins()

        for j in range(module.size_pop):
            module.fitness[j], _ = module.comp_fit(module.chrom[j])

        index = module.fitness.argmin()
        if (i + 1) % 10 == 0:
            # 获取当前时间戳 记录运算时间
            # timestamp = time.time()
            # formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            # print(formatted_time)
            print('第' + str(i + 1) + '代后的最短的路程: ' + str(module.fitness[index]))
            print('第' + str(i + 1) + '代后的最优路径:')
            # module.out_path(module.chrom[index])  # 显示每一步的最优路径

        # 存储每一步的最优路径及距离
        module.best_fit.append(module.fitness[index])
        module.best_path.append(module.chrom[index])

    best = module.chrom[0]
    print(best)
