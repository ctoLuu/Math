import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_excel('./handledData.xlsx')
df.drop(columns=['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
                 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
                 'CO(GT)_IAQI', 'C6H6(GT)_IAQI', 'NOx(GT)_IAQI',
                 'NO2(GT)_IAQI', 'NMHC(GT)_IAQI'],inplace=True)

df['Time'] = df['Time'].str.slice(start=0, stop=2).astype(int)
df = df[7:]
df.reset_index(drop=True, inplace=True)
print(df)

columns = ['Time', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'AQI']
df2 = pd.DataFrame(columns=columns)
flag = 0
count = 0
for index, row in df.iterrows():
    if row['Time'] % 8 == 0:
        average = df.iloc[index-7:index].mean(skipna=True)
        df2.loc[len(df2.index)] = average
print(df2)
df2.to_excel('./question4.xlsx')
plt.figure(figsize=(12, 8))
color_map = {
    4: 'blue',    # 假设4代表早上，用蓝色表示
    12: 'green',  # 假设12代表中午，用绿色表示
    20: 'red'     # 假设20代表晚上，用红色表示
}

# 应用颜色映射到DataFrame
df2['Color'] = df2['Time'].map(color_map)
count = 0
# 根据颜色分组数据，并为每组数据绘制散点图，同时指定标签
for time, group in df2.groupby('Time'):
    print(time)
    if count == 0:
        plt.scatter(group['NO2(GT)'], group['AQI'], c=color_map[time], label='Morning')
        count += 1
    elif count == 1:
        plt.scatter(group['NO2(GT)'], group['AQI'], c=color_map[time], label='Noon')
        count += 1
    elif count == 2:
        plt.scatter(group['NO2(GT)'], group['AQI'], c=color_map[time], label='Evening')
        count += 1
    # 注意：这里我们使用f'{time} ({time//4})' 作为标签，其中time//4是将时间转换为小时

# 添加图例，设置标题
plt.legend(title='Time of Day')

# 添加标题和标签
plt.title('Scatter Plot with Different Colors Based on Time')
plt.xlabel('NO2(GT)')
plt.ylabel('AQI')

# 显示图形
plt.show()