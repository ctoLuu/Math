from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 商品价格字典（使用历史单价的中位数）
item_prices = {
    '85123A': 2.95,
    '85099B': 2.08,
    '22423': 12.75,
    '47566': 4.95,
    '20725': 1.65
}

# 商品的购买意愿（预测值）
item_ratings = {
    '85123A': 143.1222,
    '85099B': 120.891205,
    '22423': 125.3497,
    '47566': 123.7592,
    '20725': 96.6850
}

# 提取商品编码列表
items = list(item_prices.keys())

# 商品价格列表
prices = [item_prices[item] for item in items]

# 商品意愿列表
ratings = [item_ratings[item] for item in items]

# 标准化购买意愿
scaler = MinMaxScaler()
ratings_scaled = scaler.fit_transform(np.array(ratings).reshape(-1, 1)).flatten()

# 预算
budget = 279.2  # 例如预算为50

# 创建优化问题
prob = LpProblem("Maximize_Purchase_Willingness", LpMaximize)

# 定义决策变量，表示每种商品的购买数量 (整数变量)
quantities = [LpVariable(f'quantity_{i}', lowBound=0, cat='Integer') for i in range(len(prices))]

# 定义二进制变量，表示是否选择某种商品
is_selected = [LpVariable(f'is_selected_{i}', cat='Binary') for i in range(len(prices))]

# 设置目标函数：最大化购买意愿得分与价格的加权总和 (购买意愿 / 价格) + 多样性奖励项
diversity_bonus = -30  # 多样性奖励因子
weighted_objective = lpSum([(ratings_scaled[i] / prices[i]) * quantities[i] + diversity_bonus * is_selected[i] for i in range(len(prices))])
prob += weighted_objective

# 添加预算约束：商品的总成本不超过预算
prob += lpSum([prices[i] * quantities[i] for i in range(len(prices))]) <= budget

# 添加商品数量上限约束：避免选择某一种商品的数量过大
max_quantity_per_item = 150  # 假设每种商品最多购买 10 件
for i in range(len(prices)):
    prob += quantities[i] <= max_quantity_per_item

# 添加约束条件：如果购买数量大于 0，则 is_selected 为 1
for i in range(len(prices)):
    prob += quantities[i] <= is_selected[i] * max_quantity_per_item

# 求解问题
prob.solve()

# 输出最优购买方案
print("最优购买方案:")
for i in range(len(prices)):
    print(f"商品 {items[i]}: 购买数量 {int(quantities[i].varValue)}")

# 检查求解状态
print("求解状态:", LpStatus[prob.status])
