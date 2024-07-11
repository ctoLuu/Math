import pandas as pd
import numpy as np

# df = pd.read_excel('./数据.xlsx', sheet_name=1)
# df['日期'] = df['日期'].apply(lambda x: x - 45382)
# time = pd.read_excel('./数据.xlsx', sheet_name=0)
# time.drop(columns=['配送中心', '允许到店时间段'], inplace=True)
# time['时间属性'] = time['时间属性'].map({'夜配':0, '日配':1})
# df['配送时间'] = 0
#
# for i in df.index:
#     filt = (time['到达门店简称'] == df.loc[i, '门店名称'])
#     print(time.loc[filt, '时间属性'])
#     df.loc[i, '配送时间'] = time.loc[filt, '时间属性'].iloc[0]
# print(df)
# df.to_excel('./time_send.xlsx', index=False)