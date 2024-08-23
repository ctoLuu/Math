from math import floor
import numpy as np
import time
import matplotlib.pyplot as plt  # 导入所需要的库
from pathlib import Path

from test import heuristic, a_star_search
def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    # #agents
    line = f.readline()
    num_agents = int(line)
    # #agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    return my_map, starts, goals
class GA(object):
    def __init__(self, map, starts, goals,
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

        self.goals = goals
        self.map = map
        self.starts = starts
        self.num = len(starts)  # 城市个数 对应染色体长度
        self.heuristics = []

        # 通过选择概率确定子代的选择个数
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)

        # 父代和子代群体的初始化（不直接用np.zeros是为了保证单个染色体的编码为整数，np.zeros对应的数据类型为浮点型）
        self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop,
                                                                      self.num)  # 父 print(chrom.shape)(200, 14)
        self.sub_sel = np.array([0] * int(self.select_num) * self.num).reshape(self.select_num, self.num)  # 子 (160, 14)

        # 存储群体中每个染色体的路径总长度，对应单个染色体的适应度就是其倒数  #print(fitness.shape)#(200,)
        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []  ##最优距离
        self.best_path = []  # 最优路径

    def rand_chrom(self):
        rand_ch = np.array(range(self.num))
        for i in range(self.size_pop):
            np.random.shuffle(rand_ch)
            self.chrom[i, :] = rand_ch
            self.fitness[i] = self.comp_fit(rand_ch)

    def comp_fit(self, one_path):
        time = 0
        path, res = a_star_search(self.map, (0, 0), self.starts[one_path[0]])
        time += res
        for i in range(self.num * 2 - 1):
            if i % 2 == 0:
                path, res = a_star_search(self.map, self.starts[one_path[int(i / 2)]], self.goals[one_path[int(i / 2)]])
            else:
                path, res = a_star_search(self.map, self.goals[one_path[int(i / 2)]], self.starts[one_path[int(i / 2 + 1)]])
            time += res
        return time

    def select_sub(self):
        fit = 1. / (self.fitness)  # 适应度函数
        cumsum_fit = np.cumsum(fit)  # 累积求和   a = np.array([1,2,3]) b = np.cumsum(a) b=1 3 6
        pick = cumsum_fit[-1] / self.select_num * (
                np.random.rand() + np.array(range(int(self.select_num))))  # select_num  为子代选择个数 160
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]  # chrom 父

    # 交叉，依概率对子代个体进行交叉操作
    def cross_sub(self):
        if self.select_num % 2 == 0:  # select_num160
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num + 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1, :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i + 1, :])

    def intercross(self, ind_a, ind_b):  # ind_a，ind_b 父代染色体 shape=(1,14) 14=14个城市
        r1 = np.random.randint(self.num)  # 在num内随机生成一个整数 ，num=14.即随机生成一个小于14的数
        r2 = np.random.randint(self.num)
        while r2 == r1:  # 如果r1==r2
            r2 = np.random.randint(self.num)  # r2重新生成
        left, right = min(r1, r2), max(r1, r2)  # left 为r1,r2小值 ，r2为大值
        ind_a1 = ind_a.copy()  # 父亲
        ind_b1 = ind_b.copy()  # 母亲
        for i in range(left, right + 1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[i] = ind_b1[i]  # 交叉 （即ind_a  （1,14） 中有个元素 和ind_b互换
            ind_b[i] = ind_a1[i]
            x = np.argwhere(ind_a == ind_a[i])
            y = np.argwhere(ind_b == ind_b[i])

            """
                   下面的代码意思是 假如 两个父辈的染色体编码为【1234】，【4321】 
                   13421 23412
                   交叉后为【1334】，【4221】
                   交叉后的结果是不满足条件的，重复个数为2个
                   需要修改为【1234】【4321】（即修改会来
            """
            if len(x) == 2:
                ind_a[x[x != i]] = ind_a2[i]  # 查找ind_a 中元素=- ind_a[i] 的索引
            if len(y) == 2:
                ind_b[y[y != i]] = ind_b2[i]
        return ind_a, ind_b

    # 变异模块  在变异概率的控制下，对单个染色体随机交换两个点的位置。
    def mutation_sub(self):
        for i in range(int(self.select_num)):  # 遍历每一个 选择的子代
            if np.random.rand() <= self.pmuta_prob:  # 如果随机数小于变异概率
                r1 = np.random.randint(self.num)  # 随机生成小于num==可设置 的数
                r2 = np.random.randint(self.num)
                while r2 == r1:  # 如果相同
                    r2 = np.random.randint(self.num)  # r2再生成一次
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i, [r2, r1]]  # 随机交换两个点的位置。

    # 进化逆转  将选择的染色体随机选择两个位置r1:r2 ，将 r1:r2 的元素翻转为 r2:r1 ，如果翻转后的适应度更高，则替换原染色体，否则不变
    def reverse_sub(self):
        for i in range(int(self.select_num)):  # 遍历每一个 选择的子代
            r1 = np.random.randint(self.num)  # 随机生成小于num==14 的数
            r2 = np.random.randint(self.num)
            while r2 == r1:  # 如果相同
                r2 = np.random.randint(self.num)  # r2再生成一次
            left, right = min(r1, r2), max(r1, r2)  # left取r1 r2中小值，r2取大值
            sel = self.sub_sel[i, :].copy()  # sel 为父辈染色体 shape=（1,14）

            sel[left:right + 1] = self.sub_sel[i, left:right + 1][::-1]  # 将染色体中(r1:r2)片段 翻转为（r2:r1)
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i, :]):  # 如果翻转后的适应度小于原染色体，则不变
                self.sub_sel[i, :] = sel

    # 子代插入父代，得到相同规模的新群体
    def reins(self):
        index = np.argsort(self.fitness)[::-1]  # 替换最差的（倒序）
        self.chrom[index[:self.select_num], :] = self.sub_sel

    def info(self, one_path):
        Path = []
        time = 0
        path, res = a_star_search(self.map, (0, 0), self.starts[one_path[0]])
        Path.append(path)
        time += res
        for i in range(self.num * 2 - 1):
            if i % 2 == 0:
                path, res = a_star_search(self.map, self.starts[one_path[int(i / 2)]], self.goals[one_path[int(i / 2)]])
            else:
                path, res = a_star_search(self.map, self.goals[one_path[int(i / 2)]],
                                          self.starts[one_path[int(i / 2 + 1)]])
            time += res
            Path.append(path)
        a = Path
        a = [a[i][1:] if i != 0 else a[i][:] for i in range(len(a))]
        b = []
        for z in a:
            for i in z:
                b.append(i)
        return b


if __name__ == '__main__':
    my_map, starts, goals = import_mapf_instance("./instances/16x16map.txt")
    print(my_map)
    module = GA(my_map, starts, goals)
    module.rand_chrom()
    for i in range(module.maxgen):
        module.select_sub()  # 选择子代
        module.cross_sub()  # 交叉
        module.mutation_sub()  # 变异
        module.reverse_sub()  # 进化逆转
        module.reins()  # 子代插入

        for j in range(module.size_pop):
            module.fitness[j] = module.comp_fit(module.chrom[j])

        index = module.fitness.argmin()
        if (i + 1) % 10 == 0:
            print('第' + str(i + 1) + '代后的最短的路程: ' + str(module.fitness[index]))
            print('第' + str(i + 1) + '代后的最优路径:')

            print(module.chrom[index])
            path = module.info(module.chrom[index])
            print(path)

        # 存储每一步的最优路径及距离
        module.best_fit.append(module.fitness[index])
        module.best_path.append(module.chrom[index])


    print(module.best_path[module.size_pop - 1])

