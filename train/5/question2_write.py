import numpy as np
import pandas as pd

df = pd.read_excel('./预测结果_单柜_2024-03-18.xlsx')

array =

sum = 0
for i in range(5760):
    if i < 1920:
        if sum + df.loc[i, 'FH'] < 215 * 240:
            df.loc[i, 'CN_问题2'] = df.loc[i, 'FH']
            sum += df.loc[i, 'CN_问题2']
        elif sum + df.loc[i, 'FH'] == 215 * 240:
            df.loc[i, 'CN_问题2'] = sum + df.loc[i, 'FH'] - 215 * 240
            sum +=
    elif i < 2640:

    elif i < 3140: