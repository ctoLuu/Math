import pandas as pd
import numpy as np
import time
from math import floor
import random
from copy import deepcopy

class GA(object):
    def __init__(self, time_matrix, send_data, maxgen=1000, size_pop=50000, cross_prob=0.80, pmuta_prob=0.02, select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.time_matrix = time_matrix
        self.send_data = send_data
        self.day_num = [20, 27, 27, 24]
        self.day1_node = list((self.send_data.get_group((1, ))['门店名称']).to_numpy())
        self.day2_node = list((self.send_data.get_group((2, ))['门店名称']).to_numpy())
        self.day3_node = list((self.send_data.get_group((3, ))['门店名称']).to_numpy())
        self.day4_node = list((self.send_data.get_group((4, ))['门店名称']).to_numpy())
        self.day1_time = list((self.send_data.get_group((1, ))['配送时间']).to_numpy())
        self.day2_time = list((self.send_data.get_group((2, ))['配送时间']).to_numpy())
        self.day3_time = list((self.send_data.get_group((3, ))['配送时间']).to_numpy())
        self.day4_time = list((self.send_data.get_group((4, ))['配送时间']).to_numpy())

        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        self.chrom = []
        self.sub_sel = []
        self.fitness = np.zeros(self.size_pop)
        self.best_fit = []
        self.best_path = []

    def rand_chrom(self):
        part1 = np.array(list(range(20)))
        part2 = np.array(list(range(20, 47)))
        part3 = np.array(list(range(47, 74)))
        part4 = np.array(list(range(74, 98)))
        for i in range(self.size_pop):
            np.random.shuffle(part1)
            np.random.shuffle(part2)
            np.random.shuffle(part3)
            np.random.shuffle(part4)
            rand_ch = np.concatenate((part1, part2, part3, part4))
            rand_ch = self.encode(rand_ch)

    def encode(self, array):
        encode_array = np.insert(array, 0, 0)
        current_time = 0
        index = 0
        while index != len(encode_array) - 1:
            next_time = time_matrix[encode_array[index], encode_array[index + 1]]
            current_time += next_time
            if current_time > 8:
                current_time = 0
                encode_array = np.insert(encode_array, index + 1, 0)
            index += 1

        current_weight = 0
        index = 0
        while index != len(encode_array) - 1:
            next_weight =



    def decode(self, array):
        decode_array = array[array > 0]
        return decode_array

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

    send_data = pd.read_excel('./time_send.xlsx')
    for i in send_data.index:
        filt = (df['到达门店简称'] == send_data.loc[i, '门店名称'])
        send_data.loc[i, '门店名称'] = df.loc[filt, '到达门店简称'].index[0]
    send_data.index = send_data.index.apply(lambda x: x + 1)


    module = GA(time_matrix, send_data)
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