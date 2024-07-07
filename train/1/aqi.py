import pandas as pd
import numpy as np

# 读取CSV文件
file_path = './handledData3.csv'  # 根据你的实际文件路径修改
df = pd.read_csv(file_path, delimiter=';', decimal=',')

print(df)
# 删除不需要分析的列
columns_to_drop = ['Unnamed: 15', 'Unnamed: 16']
df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# 将剩余的列转换为数值类型
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

def calculate_IAQI(value, pollutant):
    if pollutant == 'CO(GT)':
        if 0 <= value < 2:
            return ((50 - 0) / (2 - 0)) * (value - 0) + 0
        elif 2 <= value < 10:
            return ((100 - 51) / (10 - 2)) * (value - 2) + 51
        elif 10 <= value < 35:
            return ((150 - 101) / (35 - 10)) * (value - 10) + 101
        elif 35 <= value < 60:
            return ((200 - 151) / (60 - 35)) * (value - 35) + 151
        elif 60 <= value < 90:
            return ((300 - 201) / (90- 60)) * (value - 60) + 201
        elif 90 <= value < 120:
            return ((400 - 301) / (120 - 90)) * (value - 90) + 301
        else:
            return 400

    elif pollutant == 'C6H6(GT)':
            if 0 <= value < 15:
                return ((50 - 0) / (15 - 0)) * (value - 0) + 0
            elif 15<= value < 30:
                return ((100 - 51) / (30 - 15)) * (value - 15) + 51
            elif 30 <= value < 60:
                return ((150 - 101) / (60 - 30)) * (value - 30) + 101
            elif 60 <= value < 90:
                return ((200 - 151) / (90 - 60)) * (value - 60) + 151
            elif 90<= value < 120:
                return ((300 - 201) / (120 - 90)) * (value - 90) + 201
            elif 120 <= value < 150:
                return ((400 - 301) / (150 - 120)) * (value - 120) + 301
            else:
                return 400

    elif pollutant == 'NOx(GT)':
            if 0 <= value < 250:
                return ((50 - 0) / (250 - 0)) * (value - 0) + 0
            elif 250 <= value < 500:
                return ((100 - 51) / (500 - 250)) * (value - 250) + 51
            elif 500 <= value < 1400:
                return ((150 - 101) / (1400 - 500)) * (value - 500) + 101
            elif 1400 <= value < 2400:
                return ((200 - 151) / (2400 - 1400)) * (value - 1400) + 151
            elif 2400 <= value < 4600:
                return ((300 - 201) / (4600 - 2400)) * (value - 2400) + 201
            elif 4600 <= value < 6000:
                return ((400 - 301) / (6000 - 4600)) * (value - 4600) + 301
            else:
                return 400

    elif pollutant == 'NO2(GT)':
            if 0 <= value < 100 :
                return ((50 - 0) / (100 - 0)) * (value - 0) + 0
            elif 100 <= value < 200:
                return ((100 - 51) / (200 - 100)) * (value - 100) + 51
            elif 200 <= value < 700:
                return ((150 - 101) / (700- 200)) * (value - 200) + 101
            elif 700 <= value < 1200:
                return ((200 - 151) / (1200 - 700)) * (value - 700) + 151
            elif 1200<= value < 2340 :
                return ((300 - 201) / (2340- 1200)) * (value - 1200) + 201
            elif 2340 <= value < 3090:
                return ((400 - 301) / (3090 - 2340)) * (value - 2340) + 301
            else:
                return 400

    elif pollutant == 'NMHC(GT)':

            if 0 <= value < 200:
                return ((50 - 0) / (200 - 0)) * (value - 0) + 0
            elif 200 <= value < 350:
                return ((100 - 51) / (350 - 200)) * (value - 200) + 51
            elif 350 <= value < 500:
                return ((150 - 101) / (500 - 350)) * (value - 350) + 101
            elif 500 <= value < 700:
                return ((200 - 151) / (700 - 500)) * (value - 500) + 151
            elif 700 <= value < 900:
                return ((300 - 201) / (900 - 700)) * (value - 700) + 201
            elif 900 <= value < 1100:
                return ((400 - 301) / (1100 - 900)) * (value - 900) + 301
            else:
                return 400
    else:
        return np.nan

for index, row in df.iterrows():
        # 计算每一列的IAQI值并写入新的列
        df.loc[index, 'CO_IAQI'] = calculate_IAQI(row['CO(GT)'], 'CO(GT)')
        df.loc[index, 'C6H6_IAQI'] = calculate_IAQI(row['C6H6(GT)'], 'C6H6(GT)')
        df.loc[index, 'NOx_IAQI'] = calculate_IAQI(row['NOx(GT)'], 'NOx(GT)')
        df.loc[index, 'NO2_IAQI'] = calculate_IAQI(row['NO2(GT)'], 'NO2(GT)')
        df.loc[index, 'NMHC_IAQI'] = calculate_IAQI(row['NMHC(GT)'], 'NMHC(GT)')


# 逐行计算每个污染物的IAQI值
for pollutant in ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'NMHC(GT)']:
    iaqi_col = pollutant + '_IAQI'
    df[iaqi_col] = df[pollutant].apply(lambda x: calculate_IAQI(x, pollutant) if pd.notna(x) else 0)

# 计算AQI
def calculate_AQI(row):
    # 获取所有IAQI值
    iaqis = row[['CO_IAQI', 'C6H6_IAQI', 'NOx_IAQI', 'NO2_IAQI', 'NMHC_IAQI']]
    # 计算有效IAQI的数量
    valid_iaqis = iaqis.dropna()
    if len(valid_iaqis) >=2:
        return valid_iaqis.max()
    else:
        return np.nan

df['AQI'] = df.apply(calculate_AQI, axis=1)

# 存储修改后的数据到CSV
df.to_csv(file_path, index=False, sep=';', decimal=',')
