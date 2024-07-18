import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei']


file_path = 'handledData.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')


factor1_data = data[['six_bill_avg_amt', 'consume_amt_session12', 'consume_amt_session6', 'consume_amt_session3', 'consume_amt_session', 'six_cycle_mp_avg_amt']]
factor2_data = data[['six_bill_low_repay_num']]
factor3_data = data[['this_bill_rate', 'month_avg_use_year', 'month_avg_use_month6', 'month_avg_use_month3']]
factor4_data = data[['consume_num_session12', 'consume_num_session6', 'consume_num_session3', 'consume_num_session', 'six_bill_num']]
factor5_data = data[['six_bill_avg_debt_rate']]
factor6_data = data[['xaccount_age', 'six_cycle_mp_num']]

factor2_data = -factor2_data
factor5_data = -factor5_data


def apply_pca(data):
    pca = PCA(n_components=1)
    transformed_data = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_[0]
    components = pca.components_[0]
    return transformed_data, explained_variance, components

factor1_data_pca, factor1_explained_variance, factor1_components = apply_pca(factor1_data)
factor2_data_pca, factor2_explained_variance, factor2_components = apply_pca(factor2_data)
factor3_data_pca, factor3_explained_variance, factor3_components = apply_pca(factor3_data)
factor4_data_pca, factor4_explained_variance, factor4_components = apply_pca(factor4_data)
factor5_data_pca, factor5_explained_variance, factor5_components = apply_pca(factor5_data)
factor6_data_pca, factor6_explained_variance, factor6_components = apply_pca(factor6_data)

print(factor1_data_pca.shape)
all_factors_data_pca = np.hstack([factor1_data_pca, factor2_data_pca, factor3_data_pca, factor4_data_pca, factor5_data_pca, factor6_data_pca])

scaler = MinMaxScaler()
standardized_data = scaler.fit_transform(all_factors_data_pca)

m, n = standardized_data.shape
p = standardized_data / standardized_data.sum(axis=0)
entropy = -np.sum(p * np.log(p + 1e-10), axis=0) / np.log(m)

d = 1 - entropy
weights = d / d.sum()

factor_columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5', 'factor6']
weight_dict = {factor_columns[i]: weights[i] for i in range(len(factor_columns))}
print("各个指标的权重:")
for factor, weight in weight_dict.items():
    print(f"{factor}: {weight:.4f}")


weighted_data = standardized_data * weights


ideal_solution = weighted_data.max(axis=0)
negative_ideal_solution = weighted_data.min(axis=0)


distance_to_ideal = np.sqrt(((weighted_data - ideal_solution) ** 2).sum(axis=1))
distance_to_negative_ideal = np.sqrt(((weighted_data - negative_ideal_solution) ** 2).sum(axis=1))


scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
data['score'] = scores


data['score'].fillna(0, inplace=True)


data['normalized_score'] = scaler.fit_transform(data[['score']])


plt.hist(data['normalized_score'], bins=10, edgecolor='black')
plt.xlabel('客户综合价值得分')
plt.ylabel('客户数量')
plt.title('客户综合价值评分直方图')
plt.show()


data['user_category'] = pd.cut(data['normalized_score'], bins=[0, 0.5, 0.8, 1], labels=['普通用户', '白银用户', '黄金用户'])




user_counts = data['user_category'].value_counts()
user_ratios = data['user_category'].value_counts(normalize=True)


print("各类用户数量:")
print(user_counts)
print("\n各类用户占比:")
print(user_ratios)


plt.figure(figsize=(8, 6))
plt.pie(user_counts, labels=user_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'silver', 'gold', 'lightgrey'])
plt.title('各类用户分布图')
plt.axis('equal')
plt.show()