import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

df = pd.read_excel('./附件.xlsx', sheet_name=0)
df1 = pd.read_excel('./附件.xlsx', sheet_name=1)

df1.fillna(0.04, inplace=True)

for i in range(69):
    log_data = np.log(list(df1.iloc[i].values[1:-1].astype(float)))
    mean_log_data = np.mean(log_data, axis=0)
    centered_log_data = log_data - mean_log_data
    var_log_data = np.var(centered_log_data, axis=0)
    normalized_log_data = centered_log_data / np.sqrt(var_log_data)
    print("标准化后的数据：")
    print(normalized_log_data)