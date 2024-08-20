import pandas as pd
import numpy as np
from math import floor
import time
import matplotlib.pyplot as plt


class GA(object):
    def __init__(self, FH,
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

        self.FH = FH  # 城市的左边数据
        self.num = 480  # 城市个数 对应染色体长度

        # 通过选择概率确定子代的选择个数
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)

        # 父代和子代群体的初始化（不直接用np.zeros是为了保证单个染色体的编码为整数，np.zeros对应的数据类型为浮点型）
        self.chrom = np.zeros(self.size_pop * self.num).reshape(self.size_pop,
                                                                      self.num)  # 父 print(chrom.shape)(200, 14)
        self.sub_sel = np.zeros(self.select_num * self.num).reshape(self.select_num, self.num)  # 子 (160, 14)

        # 存储群体中每个染色体的路径总长度，对应单个染色体的适应度就是其倒数  #print(fitness.shape)#(200,)
        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []  ##最优距离
        self.best_path = []  # 最优路径

    def rand_chrom(self):
        for i in range(self.size_pop):
            rand_ch = np.zeros(self.num)
            for j in range(self.num):
                random_number = np.random.uniform(260, 308.1)
                rand_ch[j] = random_number
            self.chrom[i, :] = rand_ch
            self.fitness[i] = self.comp_fit(rand_ch)

    def comp_fit(self, CN):
        res = 581.81604
        SD = self.FH + CN
        max = 0
        for i in range(480 - 15):
            demand = 0
            for j in range(i, i + 15):
                demand += SD[j]
            demand /= 15
            if demand > max:
                max = demand

        charge = np.sum(CN) / 240
        res = res - 0.003 * 48 * max / 1.05 + (charge + 28.8) * 0.9442
        return res

    def select_sub(self):
        fit = self.fitness  # 适应度函数
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
        self.sub_sel = self.chrom[index, :]

    def cross_sub(self):
        if self.select_num % 2 == 0:  # select_num160
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num + 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1, :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i + 1, :])

    def intercross(self, ind_a, ind_b):
        r1 = np.random.randint(self.num)
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
            if self.comp_fit(sel) > self.comp_fit(self.sub_sel[i, :]):  # 如果翻转后的适应度小于原染色体，则不变
                self.sub_sel[i, :] = sel

    # 子代插入父代，得到相同规模的新群体
    def reins(self):
        index = np.argsort(self.fitness)[::]  # 替换最差的（倒序）
        self.chrom[index[:self.select_num], :] = self.sub_sel

    def info(self, CN):
        res = 581.81604
        SD = self.FH + CN
        max = 0
        for i in range(480 - 15):
            demand = 0
            for j in range(i, i + 15):
                demand += SD[j]
            demand /= 15
            if demand > max:
                max = demand

        charge = np.sum(CN) / 240
        res = res - 0.003 * 48 * max / 1.05 + (charge + 28.8) * 0.9442
        demand_price = 48 * max / 1.05
        charge_price = 581.81604 + (charge + 28.8) * 0.9442
        return res, demand_price, charge_price

if __name__ == "__main__":
    df = pd.read_excel('./预测结果_三并柜_2023-08-19.xlsx')
    FH = df['FH_预测'].values[2640:3120]
    print(FH)
    module = GA(FH)
    module.rand_chrom()
    for i in range(module.maxgen):
        module.select_sub()
        module.cross_sub()
        module.mutation_sub()
        module.reverse_sub()
        module.reins()

        for j in range(module.size_pop):
            module.fitness[j] = module.comp_fit(module.chrom[j])

        index = module.fitness.argmax()
        if (i + 1) % 10 == 0:
            print('第' + str(i + 1) + '代后的最短的路程: ' + str(module.fitness[index]))
            print('第' + str(i + 1) + '代后的最优路径:')
            print(module.chrom[index])
            res, demand_price, charge_price = module.info(module.chrom[index])
            print(res)
            print(demand_price)
            print(charge_price)

        module.best_fit.append(module.fitness[index])
        module.best_path.append(module.chrom[index])

    best = module.chrom[0]
    print(best)
    _, demand_price, charge_price = module.info(best)
    print(demand_price)
    print(charge_price)