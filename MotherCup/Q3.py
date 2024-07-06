import pandas as pd
from deap import base, creator, tools, algorithms
import random
# 载入数据
data = pd.read_csv('结果表2-final.csv', encoding='GB2312')
data['日期'] = pd.to_datetime(data['日期'])  # 确保日期列是正确的日期时间格式
data = data.set_index(['日期', '小时'])
data.sort_index(inplace=True)  # 确保索引排序正确

# 检查并解决重复索引的问题
if not data.index.is_unique:
    # 如果需要，可以对数据进行聚合或者删除重复
    data = data[~data.index.duplicated(keep='first')]  # 保留第一个重复项

# 提取日期和小时信息
days = data.index.get_level_values('日期').unique()
hours = range(24)

# 初始化遗传算法相关设置
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 60)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(hours))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 定义评估函数
def evalStaffing(individual):
    total_staff = sum(individual)
    efficiency_loss = 0
    for day in days:
        for i, hour in enumerate(hours):
            try:
                day_load = data.loc[(day, hour), '货量']
                if isinstance(day_load, pd.Series):
                    day_load = day_load.item()  # 确保提取单个值
                efficiency_loss += abs(25 * individual[i] - day_load)
            except KeyError:
                continue  # 如果指定的日期和小时没有数据，则跳过
    return total_staff + efficiency_loss,

toolbox.register("evaluate", evalStaffing)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=60, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
population = toolbox.population(n=300)
NGEN = 40
CXPB = 0.5
MUTPB = 0.2
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# 打印最优个体
top10 = tools.sortNondominated(population, k=10, first_front_only=True)[0]
for ind in top10:
    print(ind, ind.fitness.values)
