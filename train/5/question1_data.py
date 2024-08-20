import pandas as pd
import numpy as np

df_his = pd.read_csv('./历史用电数据_三并柜.csv')

df_his['date'] = pd.to_datetime(df_his['date'])

start_date = pd.to_datetime('2023-5-21')
end_date = df_his['date'].max()
all_dates = pd.date_range(start=start_date, end=end_date)

missing_dates = all_dates.difference(df_his['date'])
available_dates = all_dates.intersection(df_his['date'])
print("缺失的日期:")
print(missing_dates)

time_frames = []
for date in available_dates:
    start_time = pd.to_datetime(date)
    end_time = pd.to_datetime(date) + pd.DateOffset(days=1) - pd.Timedelta(seconds=15)
    time_frame = pd.date_range(start=start_time, end=end_time, freq='15s')
    time_frames.append(time_frame)

full_time_series = pd.concat([pd.Series(times) for times in time_frames]).reset_index(drop=True)
print(full_time_series)
df_expand = df_his['data'].str.split(',', expand=True)

df = pd.DataFrame()
df.index.name = 'Index'
df['Time'] = np.random.random(518400)
df['CN'] = np.random.random(518400)
df['SD'] = np.random.random(518400)
print(df)
df_transposed_CN = pd.DataFrame()
df_transposed_SD = pd.DataFrame()
for i in range(270):
    if i % 3 == 1:
        df_transpose = df_expand.loc[i].transpose()
        df_transposed_CN = pd.concat((df_transposed_CN, df_transpose), ignore_index=True)
    if i % 3 == 2:
        df_transpose = df_expand.loc[i].transpose()
        df_transposed_SD = pd.concat((df_transposed_SD, df_transpose), ignore_index=True)

df.loc[:, 'Time'] = full_time_series.values
df.loc[:, 'CN'] = df_transposed_CN.values
df.loc[:, 'SD'] = df_transposed_SD.values

print(df)
df.to_excel('./origin_data_three.xlsx')
