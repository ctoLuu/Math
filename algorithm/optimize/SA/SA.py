import numpy as np

def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def total_distance(order, cities):
    total = 0
    for i in range(len(order) - 1):
        total += distance(cities[order[i]], cities[order[i + 1]])
    return total + distance(cities[order[-1]], cities[order[0]])

def simulated_annealing(cities, initial_order, temperature, cooling_rate):
    current_order = initial_order
    best_order = current_order
    while temperature > 1e-5:
        new_order = np.random.permutation(current_order)
        current_distance = total_distance(current_order, cities)
        new_distance = total_distance(new_order, cities)
        if new_distance < current_distance or np.random.rand() < np.exp((current_distance - new_distance) / temperature):
            current_order = new_order
        if total_distance(current_order, cities) < total_distance(best_order, cities):
            best_order = current_order
        temperature *= cooling_rate
    return best_order

# 示例
np.random.seed(42)
num_cities = 24
cities = np.array([20.00, 96.10, 16.47, 94.44,
                   20.09, 92.54, 22.39, 93.37,
                   25.23, 97.24, 22.00, 96.05,
                   20.47, 97.02, 17.20, 96.29,
                   16.30, 97.38, 25.05, 98.12,
                   16.53, 96.50, 21.52, 95.59,
                   19.41, 97.13, 20.09, 92.55,
                   20.10, 95.10, 20.20, 94.10,
                   20.60, 96.88, 18.66, 90.20,
                   21.50, 94.50, 19.00, 94.00,
                   22.00, 92.00, 24.00, 93.00,
                   23.00, 95.50, 19.50, 95.50]).reshape(24, 2)
initial_order = np.arange(num_cities)
np.random.shuffle(initial_order)

final_order = simulated_annealing(cities, initial_order, temperature=1000, cooling_rate=0.995)
print("最优解的顺序:", final_order)
print("最优解的总距离:", total_distance(final_order, cities))