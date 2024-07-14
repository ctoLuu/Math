import pandas as pd
import numpy as np
import time
from math import floor
import random
from copy import deepcopy

class GA(object):
    def __init__(self, time_matrix, distance_matrix, send_data, send_array, maxgen=1000, size_pop=100, cross_prob=0.80, pmuta_prob=0.02, select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.time_matrix = time_matrix
        self.distance_matrix = distance_matrix
        self.send_data = send_data
        self.send_array = send_array
        self.day_num = [20, 27, 27, 24]

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        self.chrom = []
        self.sub_sel = []
        self.fitness = np.zeros(self.size_pop)
        self.best_fit = []
        self.best_path = []

    def rand_chrom(self):
        part1 = np.array(list(range(1, 21)))
        part2 = np.array(list(range(21, 48)))
        part3 = np.array(list(range(48, 75)))
        part4 = np.array(list(range(75, 99)))
        for i in range(self.size_pop):
            np.random.shuffle(part1)
            np.random.shuffle(part2)
            np.random.shuffle(part3)
            np.random.shuffle(part4)
            rand_ch = np.concatenate((part1, part2, part3, part4))
            new_rand_ch = self.encode(rand_ch)
            self.fitness[i] = self.comfit(new_rand_ch)
            self.chrom.append(new_rand_ch)

    def simple_encode(self, array):
        encode_array = np.insert(array, 0, 0)
        current_time = 0
        index = 0
        while index != len(encode_array) - 1:
            if self.send_data.loc[encode_array[index], '门店名称'] != self.send_data.loc[encode_array[index + 1], '门店名称']:
                next_time = time_matrix[self.send_data.loc[encode_array[index], '门店名称'], self.send_data.loc[encode_array[index + 1], '门店名称']]
            else:
                next_time = 0
            if current_time > 8:
                current_time = 0
                encode_array = np.insert(encode_array, index + 1, 0)
            index += 1

        current_weight = 0
        index = 0
        while index != len(encode_array) - 1:
            if encode_array[index] == 0:
                current_weight = 0
                index += 1
                continue
            next_weight = send_data.loc[encode_array[index+1], '总货量']
            current_weight += next_weight
            if current_weight > 3:
                current_weight = 0
                encode_array = np.insert(encode_array, index + 1, 0)
                index += 1
                continue
            index += 1
        return encode_array

    def encode(self, array):
        encode_array = np.insert(array, 0, 0)
        current_time = 0
        index = 0
        while index != len(encode_array) - 1:
            if self.send_data.loc[encode_array[index], '门店名称'] != self.send_data.loc[encode_array[index+1], '门店名称']:
                next_time = time_matrix[self.send_data.loc[encode_array[index], '门店名称'], self.send_data.loc[encode_array[index+1], '门店名称']]
            else:
                next_time = 0
            current_time += next_time
            if current_time > 8:
                current_time = 0
                encode_array = np.insert(encode_array, index + 1, 0)
            index += 1

        current_weight = 0
        index = 0
        while index != len(encode_array) - 1:
            if encode_array[index] == 0:
                current_weight = 0
                index += 1
                continue
            next_weight = self.send_data.loc[encode_array[index+1], '总货量']
            current_weight += next_weight
            if current_weight > 3:
                current_weight = 0
                encode_array = np.insert(encode_array, index + 1, 0)
                index += 1
                continue
            index += 1

        sub_arrays = [[], [], [], []]
        flag = 0
        index = 0
        for i in range(len(encode_array)):
            if encode_array[i] == 0:
                sub_arrays[flag].append(0)
            elif encode_array[i] < 21:
                sub_arrays[0].append(encode_array[i])
                flag = 0
            elif encode_array[i] < 48:
                sub_arrays[1].append(encode_array[i])
                flag = 1
            elif encode_array[i] < 75:
                sub_arrays[2].append(encode_array[i])
                flag = 2
            elif encode_array[i] < 99:
                sub_arrays[3].append(encode_array[i])
                flag = 3
        for sub_array in sub_arrays:
            sub_array = np.array(sub_array)
            zero_indices = np.where(sub_array == 0)[0]
            arrs = []
            start_index = 0
            for i in zero_indices:
                arrs.append(sub_array[start_index:i])
                start_index = i + 1
            arrs.append(sub_array[start_index:])

            flag = 0
            new_flag = 0
            new_sub_array = []
            final = []
            for arr in arrs:
                if len(arr) == 0:
                    new_flag = 1
                    continue
                if new_flag == 0:
                    new_flag = 2
                    continue
                time_list = []
                for i in arr:
                    time_list.append(self.send_data.loc[i, '配送时间'])
                if len(time_list) < 3:
                    new_sub_array += list(arr)
                else:
                    for i in range(len(time_list) - 2):
                        if time_list[i] == 0 and time_list[i + 1] == 1 and time_list[i + 2] == 0:
                            new_arr = self.find_fit(arr, time_list)
                            new_sub_array += list(new_arr)
                            flag = 1
                            break
                        if time_list[i] == 1 and time_list[i + 1] == 0 and time_list[i + 2] == 1:
                            new_arr = self.find_fit(arr, time_list)
                            new_sub_array += list(new_arr)
                            flag = 1
                            break
                    if flag == 0:
                        if time_list.count(0) > 1 and time_list[0] == 1:
                            zero_indices = np.where(np.array(time_list) == 0)[0]
                            new_arr = self.find_fit(arr, time_list)
                            final += list(new_arr)
                            flag = 1
                    if flag == 0:
                        new_sub_array += list(arr)
                    else:
                        flag = 0
            if new_flag == 2:
                time_list = []
                for i in arrs[0]:
                    time_list.append(self.send_data.loc[i, '配送时间'])
                if time_list.count(1) > 0:
                    new_sub_array += list(arrs[0])
                else:
                    print(arrs[0])
                    arrs[0] += list(new_sub_array)
                    new_sub_array = list(arrs[0])
            new_sub_array += final
            new_sub_array = np.array(new_sub_array)
            sub_arrays[index] = new_sub_array
            index += 1
        above_array = list(sub_arrays[0]) + list(sub_arrays[1]) + list(sub_arrays[2]) + list(sub_arrays[3])
        above_array = np.array(above_array)
        above_array = self.simple_encode(above_array)
        return above_array

    def find_fit(self, array, time_list):
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
            encode_array1 = self.simple_encode(new_array1)
            encode_array2 = self.simple_encode(new_array2)
            if len(encode_array1[encode_array1 == 0]) < len(encode_array2[encode_array2 == 0]):
                return self.decode(encode_array1)
            else:

                return self.decode(encode_array2)
        else:
            encode_array1 = self.simple_encode(new_array1)
            return self.decode(encode_array1)

    def comp_time(self, array, flag):
        index = 0
        current_time = 0
        while index != len(array)-1:
            next_time = time_matrix[self.send_data.loc[array[index], '门店名称'], self.send_data.loc[array[index+1], '门店名称']]
            current_time += next_time
            index += 1
            if flag == 0 and current_time > 4:
                return False
        return current_time

    def decode(self, array):
        decode_array = array[array > 0]
        return decode_array

    def comfit(self, array):
        fixed_cost = 65 * 40 + 33 * 60
        variable_cost = 0
        car_num = 0
        current_car = 0
        freeze_weight = 0
        cold_weight = 0

        zero_indices = np.where(array == 0)[0]
        sub_arrays = []
        start_index = 0
        for index in zero_indices:
            sub_arrays.append(array[start_index:index])
            start_index = index + 1
        sub_arrays.append(array[start_index:])

        for sub_array in sub_arrays:
            if len(sub_array) == 0:
                continue
            sub_array = list(np.insert(sub_array, 0, 0))
            for arr in range(len(sub_array)-1):
                freeze_weight += send_data.loc[sub_array[arr], '冷冻发货量(吨)']
                cold_weight += send_data.loc[sub_array[arr], '冷藏发货量(吨)']
            for arr in range(len(sub_array)-1):
                cur_freeze_weight = send_data.loc[sub_array[arr], '冷冻发货量(吨)']
                cur_cold_weight = send_data.loc[sub_array[arr], '冷藏发货量(吨)']
                if sub_array[arr] != sub_array[arr+1]:
                    current_distance = self.distance_matrix(send_data.loc[sub_array[arr], '门店名称'], send_data.loc[sub_array[arr], '门店名称'])
                else:
                    current_distance = 0
                variable_cost += freeze_weight * current_distance * 0.005
                variable_cost += cold_weight * current_distance * 0.0035
                freeze_weight -= cur_freeze_weight
                cold_weight -= cur_cold_weight

        return variable_cost + fixed_cost

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



    def reins(self):
        index = np.argsort(self.fitness)[::-1]  # 替换最差的（倒序）
        for i in range(self.select_num):
            self.chrom[index[i]] = self.sub_sel[i]

if __name__ == "__main__":
    time_matrix = pd.read_excel('./time_matrix.xlsx')
    time_matrix.set_index(['到达门店简称'], inplace=True)
    time_matrix = time_matrix.to_numpy(dtype=float)

    df = pd.read_excel('./数据.xlsx', sheet_name=0)
    df.drop(columns=['配送中心', '允许到店时间段'], inplace=True)
    df['时间属性'] = df['时间属性'].map({'夜配': 0, '日配': 1})
    df2 = pd.DataFrame(['配送中心', '120.44739', '31.50353', 2]).T
    df2.columns = df.columns
    df = pd.concat([df2, df], ignore_index=True)
    df.drop(index=[3, 7, 23], inplace=True)
    df.reset_index(inplace=True)
    send_array = df['时间属性']
    send_array = send_array.to_numpy(dtype=int)

    send_data = pd.read_excel('./time_send.xlsx')
    for i in send_data.index:
        filt = (df['到达门店简称'] == send_data.loc[i, '门店名称'])
        send_data.loc[i, '门店名称'] = df.loc[filt, '到达门店简称'].index[0]
    df3 = pd.DataFrame([0, 0, 0, 0, 0]).T
    df3.columns = send_data.columns
    send_data = pd.concat([df3, send_data], ignore_index=True)
    send_data['总货量'] = send_data['冷冻发货量(吨)'] + send_data['冷藏发货量(吨)']

    distance_matrix = pd.read_excel('./distance_matrix.xlsx')
    distance_matrix.set_index(['到达门店简称'], inplace=True)
    print(distance_matrix)

    module = GA(time_matrix, distance_matrix, send_data, send_array)
    module.rand_chrom()
    for i in range(module.maxgen):
        module.select_sub()
        # module.cross_sub()
        # module.mutation_sub()
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