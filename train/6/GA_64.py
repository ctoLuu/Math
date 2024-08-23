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


def distribute_tasks(num_tasks, num_robots):
    # Step 1: 生成任务列表并打乱顺序
    tasks = np.arange(num_tasks)
    np.random.shuffle(tasks)

    # Step 2: 随机分配每个机器人分配到的任务数量
    # 生成 num_robots 个随机数，这些数之和为 num_tasks
    split_points = np.sort(np.random.choice(range(1, num_tasks), num_robots - 1, replace=False))

    # Step 3: 根据分割点将任务分配给每个机器人
    task_distribution = np.split(tasks, split_points)

    return task_distribution

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

        # 通过选择概率确定子代的选择个数
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)

        # 父代和子代群体的初始化（不直接用np.zeros是为了保证单个染色体的编码为整数，np.zeros对应的数据类型为浮点型）
        # self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop,
        #                                                               self.num)  # 父 print(chrom.shape)(200, 14)
        # self.sub_sel = np.array([0] * int(self.select_num) * self.num).reshape(self.select_num, self.num)  # 子 (160, 14)
        self.chrom = []
        self.sub_sel = []
        # 存储群体中每个染色体的路径总长度，对应单个染色体的适应度就是其倒数  #print(fitness.shape)#(200,)
        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []  ##最优距离
        self.best_path = []  # 最优路径
    def rand_chrom(self):
        for i in range(self.size_pop):
            rand_ch = distribute_tasks(50, 5)

            self.fitness[i] = self.comp_fit(rand_ch)
            self.chrom.append(rand_ch)

    def comp_fit(self, one_path):
        total_time = 0
        start_pos = [(10, 20), (0, 0), (44, 28), (16, 41), (26, 45)]
        Path = []
        for i in range(5):
            path = []
            time = 0
            sub_path, res = a_star_search(self.map, start_pos[i], self.starts[one_path[i][0]])
            time += res
            path.append(sub_path)
            for j in range(len(one_path[i]) * 2 - 1):
                if j % 2 == 0:
                    sub_path, res = a_star_search(self.map, self.starts[one_path[i][int(j / 2)]], self.goals[one_path[i][int(j / 2)]])
                else:
                    sub_path, res = a_star_search(self.map, self.goals[one_path[i][int(j / 2)]], self.starts[one_path[i][int(j / 2 + 1)]])
                time += res
                path.append(sub_path)
            newPath = [path[k][1:] if k != 0 else path[k][:] for k in range(len(path))]
            path = []
            for sub_path in newPath:
                for pos in sub_path:
                    path.append(pos)
            total_time += time
            Path.append(path)


        return total_time

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
        self.sub_sel = [self.chrom[x] for x in index]

    def cross_sub(self):
        if self.select_num % 2 == 0:
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num + 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i], self.sub_sel[i + 1] = self.intercross(self.sub_sel[i], self.sub_sel[i + 1])

    def intercross(self, ind_a, ind_b):
        offspring_a = []
        offspring_b = []

        for robot_tasks_a, robot_tasks_b in zip(ind_a, ind_b):
            num_tasks = len(robot_tasks_a)

            if num_tasks <= 1:
                offspring_a.append(robot_tasks_a.copy())
                offspring_b.append(robot_tasks_b.copy())
                continue

            # Select two distinct crossover points within the sublist
            r1, r2 = sorted(np.random.choice(range(num_tasks), 2, replace=False))

            # Create placeholders for offspring sublists
            child_a = [-1] * num_tasks
            child_b = [-1] * num_tasks

            # Copy the segment between crossover points from each parent
            child_a[r1:r2 + 1] = robot_tasks_a[r1:r2 + 1]
            child_b[r1:r2 + 1] = robot_tasks_b[r1:r2 + 1]

            # Fill in the remaining positions without duplicates
            remaining_tasks_a = [task for task in robot_tasks_b if task not in child_a]
            remaining_tasks_b = [task for task in robot_tasks_a if task not in child_b]

            # Fill child_a
            idx = (r2 + 1) % num_tasks
            for task in remaining_tasks_a:
                while child_a[idx] != -1:
                    idx = (idx + 1) % num_tasks
                child_a[idx] = task

            # Fill child_b
            idx = (r2 + 1) % num_tasks
            for task in remaining_tasks_b:
                while child_b[idx] != -1:
                    idx = (idx + 1) % num_tasks
                child_b[idx] = task

            offspring_a.append(child_a)
            offspring_b.append(child_b)

        return offspring_a, offspring_b

    def mutation_sub(self):
        for i in range(int(self.select_num)):
            if np.random.rand() <= self.pmuta_prob:
                robot_idx = np.random.randint(0, 5)
                if len(self.sub_sel[i][robot_idx]) > 1:
                    task_idx1, task_idx2 = np.random.choice(len(self.sub_sel[i][robot_idx]), 2, replace=False)
                    self.sub_sel[i][robot_idx][task_idx1], self.sub_sel[i][robot_idx][task_idx2] = \
                        self.sub_sel[i][robot_idx][task_idx2], self.sub_sel[i][robot_idx][task_idx1]

    def reverse_sub(self):
        for i in range(int(self.select_num)):
            robot_idx = np.random.randint(0, 5)
            tasks = self.sub_sel[i].copy()  # Create a copy to avoid unintended side-effects
            if len(tasks) > 1 and isinstance(tasks[robot_idx], list):
                if len(tasks[robot_idx]) > 1:
                    r1, r2 = sorted(np.random.choice(len(tasks[robot_idx]), 2, replace=False))
                    # Reverse the selected segment
                    tasks[robot_idx][r1:r2 + 1] = tasks[robot_idx][r1:r2 + 1][::-1]
                    # Compare fitness and replace if improved
                    if self.comp_fit(tasks) < self.comp_fit(self.sub_sel[i]):
                        self.sub_sel[i] = tasks

    def reins(self):
        index = np.argsort(self.fitness)[::-1]  # 替换最差的
        for i in range(self.select_num):
            self.chrom[index[i]] = self.sub_sel[i]


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
    my_map, starts, goals = import_mapf_instance("./instances/64x64map.txt")
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
            # path = module.info(module.chrom[index])
            # print(path)

        # 存储每一步的最优路径及距离
        module.best_fit.append(module.fitness[index])
        module.best_path.append(module.chrom[index])


    print(module.best_path[module.size_pop - 1])

