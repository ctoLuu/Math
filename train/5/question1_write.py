import pandas as pd
import numpy as np

df = pd.read_excel('./data_one.xlsx')

df['FH_预测'] = df['Predicted_SD'] - df['Predicted_CN']

df_write = pd.read_excel('./预测结果_单柜_2024-03-18.xlsx')
df_write['FH_预测'] = df['FH_预测'].values
print(df_write)
df_write.to_excel('预测结果_单柜_2024-03-18.xlsx')