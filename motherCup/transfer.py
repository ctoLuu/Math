import pandas as pd
from datetime import datetime, timedelta

# 假设的起始日期，这里以2013年1月1日作为起始点
start_date = datetime(2013, 8, 2)

# 读取Excel文件
df = pd.read_excel('Q3结果.xlsx')

# 转换日期列
# 假设日期列名为'日期'，且存储的是年份的第几天
df['日期'] = df['日期'].apply(lambda x: start_date + timedelta(days=x - 1))
df['日期'] = df['日期'].dt.strftime('%Y/%m/%d')  # 转换为2013/12/01格式

# 班次映射字典
shift_mapping = {
    1: '00:00-08:00',
    2: '05:00-13:00',
    3: '08:00-16:00',
    4: '12:00-20:00',
    5: '14:00-22:00',
    6: '16:00-24:00'
}

# 转换班次列
# 假设班次列名为'班次'
df['班次'] = df['班次'].map(shift_mapping)

# 查看转换后的DataFrame
print(df[['日期', '班次']])

df.to_csv('结果表5.csv', encoding='GBK',index=False)