import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def getFeatures(factor):
    scaler = StandardScaler(with_mean=False, with_std=True)
    df_scaled = scaler.fit_transform(factor)
    pca = PCA(n_components=1)
    transformed_data = pca.fit_transform(df_scaled)
    print(pca.explained_variance_ratio_)
    return transformed_data

df = pd.read_excel('./handledData.xlsx')
print(df)

factor_R = df[["last_mp_days"]]
factor_M = df[['consume_num_session12', 'consume_num_session6', 'consume_num_session3', 'consume_num_session', 'six_bill_num']]
factor_F = df[['six_bill_avg_amt', 'consume_amt_session12', 'consume_amt_session6', 'consume_amt_session3', 'consume_amt_session', 'six_cycle_mp_avg_amt', 'this_bill_rate', 'month_avg_use_year', 'month_avg_use_month6', 'month_avg_use_month3']]
factor_N = df[['six_bill_low_repay_num', 'six_bill_avg_debt_rate']]

feature_R = getFeatures(factor_R)
feature_F = getFeatures(factor_F)
feature_M = getFeatures(factor_M)
feature_N = getFeatures(factor_N)

feature1 = pd.DataFrame(feature_R, columns=['feature'])
feature2 = pd.DataFrame(feature_F, columns=['feature'])
feature3 = pd.DataFrame(feature_M, columns=['feature'])
feature4 = pd.DataFrame(feature_N, columns=['feature'])

# 计算五分位数并分箱
quantiles = feature1['feature'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
bins = [-np.inf, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], np.inf]
feature1['quintile'] = np.digitize(feature1['feature'], bins)
feature1['quintile'] = feature1['quintile'].map({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})

quantiles = feature2['feature'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
bins = [-np.inf, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], np.inf]
feature2['quintile'] = np.digitize(feature2['feature'], bins)

quantiles = feature3['feature'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
bins = [-np.inf, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], np.inf]
feature3['quintile'] = np.digitize(feature3['feature'], bins)

quantiles = feature4['feature'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
bins = [-np.inf, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], np.inf]
feature4['quintile'] = np.digitize(feature4['feature'], bins)
feature4['quintile'] = feature4['quintile'].map({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})

# 计算各指标的熵值
# def calculate_entropy(series):
#     n = len(series)
#     probabilities = series / series.sum()
#     entropy = -sum(probabilities * np.log(probabilities + 1e-10)) / np.log(n) # 避免对数为负无穷
#     return entropy
#
# entropies_R = calculate_entropy(feature1['quintile'])
# entropies_F = calculate_entropy(feature2['quintile'])
# entropies_M = calculate_entropy(feature3['quintile'])
# entropies_N = calculate_entropy(feature4['quintile'])
#
# difference_coefficients = 1 - entropies_R + 1 - entropies_F + 1 - entropies_M + 1 - entropies_N
# weights_R = (1 - entropies_R) / difference_coefficients
# weights_F = (1 - entropies_F) / difference_coefficients
# weights_M = (1 - entropies_M) / difference_coefficients
# weights_N = (1 - entropies_N) / difference_coefficients
#
# print(weights_R, weights_F, weights_M, weights_N)
f1 = feature1['feature'].to_numpy().reshape(-1, 1)
f2 = feature2['feature'].to_numpy().reshape(-1, 1)
f3 = feature3['feature'].to_numpy().reshape(-1, 1)
f4 = feature4['feature'].to_numpy().reshape(-1, 1)
all_factors_data_pca = np.hstack([f1, f2, f3, f4])
scaler = MinMaxScaler()
standardized_data = scaler.fit_transform(all_factors_data_pca)
print(all_factors_data_pca)
m, n = standardized_data.shape
p = standardized_data / standardized_data.sum(axis=0)
entropy = -np.sum(p * np.log(p + 1e-10), axis=0) / np.log(m)

d = 1 - entropy
weights = d / d.sum()

factor_columns = ['factor1', 'factor2', 'factor3', 'factor4']
weight_dict = {factor_columns[i]: weights[i] for i in range(len(factor_columns))}
print("各个指标的权重:")
for factor, weight in weight_dict.items():
    print(f"{factor}: {weight:.4f}")