import pandas as pd
import numpy as np
import time
from math import floor


class GA(object):
    def __init__(self, time_matrix, send_array, maxgen=1000, size_pop=100, cross_prob=0.80, pmuta_prob=0.02, select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.punishment = 10000000000
        self.num = 43
        self.time_matrix = time_matrix
        self.send_array = send_array

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        # self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop, self.num)
        self.chrom = np.array([np.array([0] * 43) for _ in range(self.size_pop)])
        # self.sub_sel = np.array([0] * int(self.select_num) * self.num).reshape(self.select_num, self.num)
        self.sub_sel = np.array([np.array([0] * 43) for _ in range(int(self.select_num))])

        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []
        self.best_path = []

    def rand_chrom(self):
        rand_ch = np.array(range(1, self.num + 1))
        for i in range(self.size_pop):
            np.random.shuffle(rand_ch)
            rand_ch = self.encode(rand_ch)
            self.chrom[i] = rand_ch
            self.fitness[i] = self.comp_fit(rand_ch)

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
                encode_array = np.insert(array, index, 0)
            index += 1
        return encode_array

    def comp_fit(self, array):
        res = 0
        zero_indices = np.where(array == 0)[0]
        sub_arrays = []
        start_index = 0
        for index in zero_indices:
            sub_arrays.append(array[start_index:index])
            start_index = index + 1
        sub_arrays.append(array[start_index:])

        flag = 0
        for arr in sub_arrays:
            time_list = []
            for i in arr:
                time_list.append(self.send_array[i])
            if flag:
                break
            if len(time_list) < 3:
                res += 1
            else:
                for i in range(len(time_list) - 2):
                    if time_list[i] == 0 and time_list[i + 1] == 1 and time_list[i + 2] == 0:
                        res = self.punishment
                        flag = 1
                        break
                    if time_list[i] == 1 and time_list[i + 1] == 0 and time_list[i + 2] == 1:
                        res = self.punishment
                        flag = 1
                        break
                res += 1
        return res

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
        self.sub_sel = self.chrom[index]

    def cross_sub(self):
        if self.select_num % 2 == 0:
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num + 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i], self.sub_sel[i + 1] = self.cross_func(self.sub_sel[i], self.sub_sel[i + 1])

    # def cross_fuc(self, array1, array2):


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
    print(time_matrix)
    send_array = df['时间属性']
    print(send_array)

    module = GA(time_matrix, send_array)
    module.rand_chrom()
    for i in range(module.maxgen):
        module.select_sub()
        module.cross_sub()
        module.mutation_sub()
        module.reverse_sub()
        module.reins()

        for j in range(module.size_pop):
            module.fitness[j] = module.comp_fit(module.chrom[j])

        index = module.fitness.argmin()
        if (i + 1) % 50 == 0:
            # 获取当前时间戳 记录运算时间
            timestamp = time.time()
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            print(formatted_time)
            print('第' + str(i + 1) + '代后的最短的路程: ' + str(module.fitness[index]))
            print('第' + str(i + 1) + '代后的最优路径:')
            module.out_path(module.chrom[index])  # 显示每一步的最优路径

        # 存储每一步的最优路径及距离
        module.best_fit.append(module.fitness[index])
        module.best_path.append(module.chrom[index])

    best = module.chrom[0]

