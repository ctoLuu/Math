import pandas as pd

df = pd.read_excel('./附件.xlsx', sheet_name=0)
print(df)
filt = df['类型'] == '高钾'
group1 = df.loc[filt, :]
group2 = df.loc[~filt, :]
print(group1)
print(group2)