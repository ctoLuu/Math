import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def getFeatures(factor):
    scaler = StandardScaler(with_mean=False, with_std=True)
    df_scaled = scaler.fit_transform(factor)
    pca = PCA(n_components=1)
    transformed_data = pca.fit_transform(df_scaled)
    print(pca.explained_variance_ratio_)
    return transformed_data

def calculate_entropy(series):
    n = len(series)
    probabilities = series / series.sum(axis=0)
    entropy = -sum(probabilities * np.log(probabilities + 1e-10)) / np.log(n) # 避免对数为负无穷
    return entropy

df = pd.read_excel('./handledData.xlsx')
df['last_mp_days'] = 9999 - df['last_mp_days']
factor_R = df[["last_mp_days",'xaccount_age']]
factor_F = df[['consume_num_session12', 'consume_num_session6', 'consume_num_session3',
               'consume_num_session', 'six_bill_num', 'six_cycle_mp_num', 'epp_nbr_12m'
               ]]
factor_M = df[['six_bill_avg_amt', 'consume_amt_session12', 'consume_amt_session6',
               'consume_amt_session3', 'consume_amt_session', 'six_cycle_mp_avg_amt'
               ]]
factor_N = df[['six_bill_low_repay_num', 'six_bill_avg_debt_rate']]

feature_R = getFeatures(factor_R)
feature_F = getFeatures(factor_F)
feature_M = getFeatures(factor_M)
feature_N = getFeatures(factor_N)

feature1 = pd.DataFrame(feature_R, columns=['feature'])
feature2 = pd.DataFrame(feature_F, columns=['feature'])
feature3 = pd.DataFrame(feature_M, columns=['feature'])
feature4 = pd.DataFrame(feature_N, columns=['feature'])

quantiles = feature1['feature'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
bins = [-np.inf, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], np.inf]
feature1['quintile'] = np.digitize(feature1['feature'], bins)
quantiles = feature2['feature'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
bins = [-np.inf, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], np.inf]
feature2['quintile'] = np.digitize(feature2['feature'], bins)
quantiles = feature3['feature'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
bins = [-np.inf, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], np.inf]
feature3['quintile'] = np.digitize(feature3['feature'], bins)
quantiles = feature4['feature'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
bins = [-np.inf, quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8], np.inf]
feature4['quintile'] = np.digitize(feature4['feature'], bins)
feature4['quintile'] = feature4['quintile'].map({1: 5, 2: 4,3:3, 4: 2, 5: 1})

scaler = MinMaxScaler()
feature1['feature'] = scaler.fit_transform(feature1['feature'].to_numpy().reshape(-1, 1))
feature2['feature'] = scaler.fit_transform(feature2['feature'].to_numpy().reshape(-1, 1))
feature3['feature'] = scaler.fit_transform(feature3['feature'].to_numpy().reshape(-1, 1))
feature4['feature'] = scaler.fit_transform(feature4['feature'].to_numpy().reshape(-1, 1))

entropies_R = calculate_entropy(feature1['feature'])
entropies_F = calculate_entropy(feature2['feature'])
entropies_M = calculate_entropy(feature3['feature'])
entropies_N = calculate_entropy(feature4['feature'])

difference_coefficients = 1 - entropies_R + 1 - entropies_F + 1 - entropies_M + 1 - entropies_N
weights_R = (1 - entropies_R) / difference_coefficients
weights_F = (1 - entropies_F) / difference_coefficients
weights_M = (1 - entropies_M) / difference_coefficients
weights_N = (1 - entropies_N) / difference_coefficients

part1_R = feature1['feature'].sum() * weights_R
part1_F = feature2['feature'].sum() * weights_F
part1_M = feature3['feature'].sum() * weights_M
part1_N = feature4['feature'].sum() * weights_N
part1 = part1_R + part1_F + part1_M + part1_N

weights2_R, weights2_F, weights2_M, weights2_N = 0.099, 0.345, 0.370, 0.185
part2_R = feature1['feature'].sum() * weights2_R
part2_F = feature2['feature'].sum() * weights2_F
part2_M = feature3['feature'].sum() * weights2_M
part2_N = feature4['feature'].sum() * weights2_N
part2 = part2_R + part2_F + part2_M + part2_N

T = part2 / (part1 + part2)
U = part1 / (part1 + part2)
weights3_R = weights2_R * T + weights_R * U
weights3_F = weights2_F * T + weights_R * U
weights3_M = weights2_M * T + weights_M * U
weights3_N = weights2_N * T + weights_N * U

feature1['R_S'] = feature1['quintile'] * weights3_R
feature2['F_S'] = feature2['quintile'] * weights3_F
feature3['M_S'] = feature3['quintile'] * weights3_M
feature4['N_S'] = feature4['quintile'] * weights3_N
feature = pd.concat([feature1['R_S'], feature2['F_S'], feature3['M_S'], feature4['N_S']], axis=1)
feature['S'] = feature['R_S'] + feature['F_S'] + feature['M_S'] + feature['N_S']

k_values = range(2,9)
sse_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(feature)
    sse_scores.append(kmeans.inertia_)

# 绘制Elbow图
plt.plot(k_values, sse_scores, marker='o', label='SSE')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(feature)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

clustered_data = {}
for i in range(6):  # 假设有6个簇
    clustered_data[i] = feature[kmeans.labels_ == i]
for i, cluster in clustered_data.items():
    # print(f"\nCluster {i}:")
    # print(cluster.index)
    # print(df.loc[cluster.index, :].describe())  # 211 3047 5971 7090 8576 9982
    if 36 in cluster.index:
        print(i)
        print(df.loc[36, ['id', 'cred_limit']])
        print(df.loc[36, 'cred_limit'])
    if 500 in cluster.index:

        print(i)
        print(df.loc[500, ['id', 'cred_limit']])
    if 948 in cluster.index:
        print(i)
        print(df.loc[948, ['id', 'cred_limit']])
    if 8986 in cluster.index:
        print(i)
        print(df.loc[8986, ['id', 'cred_limit']])
    if 9319 in cluster.index:
        print(i)
        print(df.loc[9319, ['id', 'cred_limit']])
    if 9580 in cluster.index:
        print(i)
        print(df.loc[9580, ['id', 'cred_limit']])



