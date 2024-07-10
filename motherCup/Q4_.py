import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 全局变量
POP_SIZE = 40
GENERATIONS = 80
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
NUM_DAYS = 30
NUM_SHIFTS = 6
NUM_EMPLOYEES = 200
FULL_TIME_EFFICIENCY = 25 * 8
TEMP_EFFICIENCY = 20 * 8

# 根据小时数指定班次
def assign_shift(hour):
    shifts = [0, 1, 2, 3, 4, 5, 6]
    return shifts[((hour - 1) // 5) % 7]

# 初始化种群
def initialize_population():
    return [np.random.randint(0, NUM_SHIFTS + 1, NUM_DAYS) for _ in range(POP_SIZE)]

# 适应度函数
def fitness(individual, demands):
    total_temp_workers = 0
    for day_demand in demands:
        required_workers = day_demand - sum(individual[assign_shift(hour)] for hour in range(NUM_DAYS))
        total_temp_workers += max(0, int(np.ceil(required_workers / TEMP_EFFICIENCY)))
    return total_temp_workers

# 交叉操作
def crossover(parent1, parent2):
    point = np.random.randint(1, NUM_DAYS)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# 变异操作
def mutate(individual):
    for i in range(NUM_DAYS):
        if np.random.rand() < MUTATION_RATE:
            individual[i] = np.random.randint(0, NUM_SHIFTS + 1)

# 运行遗传算法
def genetic_algorithm(demands):
    population = initialize_population()
    best_fitness = float('inf')
    best_individual = None

    for _ in range(GENERATIONS):
        fitness_scores = [fitness(ind, demands) for ind in population]
        best_index = np.argmin(fitness_scores)
        if fitness_scores[best_index] < best_fitness:
            best_fitness = fitness_scores[best_index]
            best_individual = population[best_index]

        new_population = []
        for _ in range(0, POP_SIZE, 2):
            parent1, parent2 = population[np.random.choice(range(POP_SIZE), 2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population[:POP_SIZE]

    return best_individual

# 加载和处理数据
def load_and_process_data(file_path):
    data = pd.read_csv(file_path, encoding='GB2312')
    sc60_data = data[data['分拣中心'] == 'SC60']
    sc60_data['班次'] = sc60_data['小时'].apply(assign_shift)
    day_demands = sc60_data.pivot_table(index='日期', columns='班次', values='货量', aggfunc='sum').fillna(0)
    return day_demands

# 可视化排班情况
def visualize_schedule(best_schedule):
    if best_schedule is not None:
        plt.figure(figsize=(15, 5))
        plt.plot(best_schedule, 'o-', color='cyan')
        plt.title('Full-time Employee Schedule Over 30 Days')
        plt.xlabel('Day')
        plt.ylabel('Shift')
        plt.xticks(range(NUM_DAYS))
        plt.yticks(range(1, NUM_SHIFTS + 1))
        plt.grid(True)
        plt.show()

# 主函数
def main():
    file_path = '结果表4.csv'
    try:
        day_demands = load_and_process_data(file_path)
        best_schedule = genetic_algorithm(day_demands)
        visualize_schedule(best_schedule)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()