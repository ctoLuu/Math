import numpy as np
import pandas as pd

df = pd.read_excel('./origin_data.xlsx')
df = df.interpolate(method='linear')

SD = df['SD'].values[593280:]
print(SD)
max = 0
for i in range(97920-15):
    demand = 0
    for j in range(i, i + 15):
        demand += SD[j]
    demand /= 15
    if demand > max:
        max = demand
print(max)