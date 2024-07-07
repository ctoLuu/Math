import pandas as pd
import numpy as np

# 读取Excel文件
file_path = './handledData4.xlsx'
df = pd.read_excel(file_path)

# 删除不需要分析的列
columns_to_drop = ['Unnamed: 15', 'Unnamed: 16']
df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# 将剩余的列转换为数值类型
for col in df.columns:
    if col == 'Date' or col == 'Time':
        continue
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 计算分指数（IAQI）
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
            return ((300 - 201) / (90 - 60)) * (value - 60) + 201
        elif 90 <= value < 120:
            return ((400 - 301) / (120 - 90)) * (value - 90) + 301
        else:
            return 0

    elif pollutant == 'C6H6(GT)':
        if 0 <= value < 15:
            return ((50 - 0) / (15 - 0)) * (value - 0) + 0
        elif 15 <= value < 30:
            return ((100 - 51) / (30 - 15)) * (value - 15) + 51
        elif 30 <= value < 60:
            return ((150 - 101) / (60 - 30)) * (value - 30) + 101
        elif 60 <= value < 90:
            return ((200 - 151) / (90 - 60)) * (value - 60) + 151
        elif 90 <= value < 120:
            return ((300 - 201) / (120 - 90)) * (value - 90) + 201
        elif 120 <= value < 150:
            return ((400 - 301) / (150 - 120)) * (value - 120) + 301
        else:
            return 0

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
            return 0

    elif pollutant == 'NO2(GT)':
        if 0 <= value < 100:
            return ((50 - 0) / (100 - 0)) * (value - 0) + 0
        elif 100 <= value < 200:
            return ((100 - 51) / (200 - 100)) * (value - 100) + 51
        elif 200 <= value < 700:
            return ((150 - 101) / (700 - 200)) * (value - 200) + 101
        elif 700 <= value < 1200:
            return ((200 - 151) / (1200 - 700)) * (value - 700) + 151
        elif 1200 <= value < 2340:
            return ((300 - 201) / (2340 - 1200)) * (value - 1200) + 201
        elif 2340 <= value < 3090:
            return ((400 - 301) / (3090 - 2340)) * (value - 2340) + 301
        else:
            return 0

    elif pollutant == 'NMHC(GT)':
        if 0 <= value < 250:
            return ((50 - 0) / (250 - 0)) * (value - 0) + 0
        elif 250 <= value < 800:
            return ((100 - 51) / (800 - 250)) * (value - 250) + 51
        elif 800 <= value < 1400:
            return ((150 - 101) / (1400 - 800)) * (value - 800) + 101
        elif 1400 <= value < 2000:
            return ((200 - 151) / (2000 - 1400)) * (value - 1400) + 151
        elif 2000 <= value < 2700:
            return ((300 - 201) / (2700 - 2000)) * (value - 2000) + 201
        elif 2700 <= value < 3700:
            return ((400 - 301) / (3700 - 2700)) * (value - 2700) + 301
        else:
            return 0

    else:
        return np.nan

# 逐行计算每个污染物的IAQI值
for pollutant in ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'NMHC(GT)']:
    iaqi_col = pollutant + '_IAQI'
    df[iaqi_col] = df[pollutant].apply(lambda x: calculate_IAQI(x, pollutant) if pd.notna(x) else np.nan)

# 计算AQI
def calculate_AQI(row):
    # 获取所有IAQI值
    iaqis = row[['CO(GT)_IAQI', 'C6H6(GT)_IAQI', 'NOx(GT)_IAQI', 'NO2(GT)_IAQI', 'NMHC(GT)_IAQI']]
    # 计算有效IAQI的数量
    valid_iaqis = iaqis.dropna()
    if len(valid_iaqis) >= 2:
        return valid_iaqis.max()
    else:
        return np.nan

df['AQI'] = df.apply(calculate_AQI, axis=1)
# 统计每个污染物的分指数在AQI中成为最大值的频率，忽略缺失值的行
contribution = {
    'CO(GT)_IAQI': ((df['CO(GT)_IAQI'] == df['AQI']) & (~df[['CO(GT)_IAQI', 'C6H6(GT)_IAQI', 'NOx(GT)_IAQI', 'NO2(GT)_IAQI', 'NMHC(GT)_IAQI']].isnull().any(axis=1))).sum(),
    'C6H6(GT)_IAQI': ((df['C6H6(GT)_IAQI'] == df['AQI']) & (~df[['CO(GT)_IAQI', 'C6H6(GT)_IAQI', 'NOx(GT)_IAQI', 'NO2(GT)_IAQI', 'NMHC(GT)_IAQI']].isnull().any(axis=1))).sum(),
    'NOx(GT)_IAQI': ((df['NOx(GT)_IAQI'] == df['AQI']) & (~df[['CO(GT)_IAQI', 'C6H6(GT)_IAQI', 'NOx(GT)_IAQI', 'NO2(GT)_IAQI', 'NMHC(GT)_IAQI']].isnull().any(axis=1))).sum(),
    'NO2(GT)_IAQI': ((df['NO2(GT)_IAQI'] == df['AQI']) & (~df[['CO(GT)_IAQI', 'C6H6(GT)_IAQI', 'NOx(GT)_IAQI', 'NO2(GT)_IAQI', 'NMHC(GT)_IAQI']].isnull().any(axis=1))).sum(),
    'NMHC(GT)_IAQI': ((df['NMHC(GT)_IAQI'] == df['AQI']) & (~df[['CO(GT)_IAQI', 'C6H6(GT)_IAQI', 'NOx(GT)_IAQI', 'NO2(GT)_IAQI', 'NMHC(GT)_IAQI']].isnull().any(axis=1))).sum(),
}

# 转换为DataFrame并排序
contribution_df = pd.DataFrame.from_dict(contribution, orient='index', columns=['Frequency'])
contribution_df = contribution_df.sort_values(by='Frequency', ascending=False)

# 打印贡献度排序
print(contribution_df)

# 保存结果到Excel文件
df.to_excel('./handledData.xlsx', index=False)