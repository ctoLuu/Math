import numpy as np
import pandas as pd

price = {
    '谷': 0.3096,
    '平': 0.688,
    '峰': 1.1353,
}

time_price = ['谷', '谷', '谷', '谷', '谷', '谷', '谷', '谷',
              '峰', '峰', '峰',
              '谷', '谷',
              '峰', '峰', '峰', '峰',
              '平', '平', '平', '平', '平', '平', '平']
time_price = [price[i] for i in time_price]

df = pd.read_excel('./predicted_results_fft_one.xlsx')
count = 0
for i in range(5760):
    current = df.loc[i, 'Predicted_CN']
    if current > 101.4:
        current = 101.4
        count += 1
        print(f'find greater than 101.4 {i}')
    elif current < -101.4:
        current = -101.4
        count += 1
        print(f'find less than -101.4 {i}')
    elif (current < 10) and (current > -10):
        df.loc[i, 'Predicted_CN'] = float(0)
        count += 1
    elif ((current > 10) and (current < 40)) or ((current > 70) and (current < 90)):
        df.loc[i, 'Predicted_CN'] = (df.loc[i-1, 'Predicted_CN'] +
                                     df.loc[i-2, 'Predicted_CN'] +
                                     df.loc[i-3, 'Predicted_CN']) / 3
        count += 1
        print(f'find middle {i}')
    elif ((current < -10) and (current > -40)) or ((current < -70) and (current > -90)):
        df.loc[i, 'Predicted_CN'] = (df.loc[i-1, 'Predicted_CN'] +
                                     df.loc[i-2, 'Predicted_CN'] +
                                     df.loc[i-3, 'Predicted_CN']) / 3
        count += 1
        print(f'find middle {i}')

df.to_excel('./data_one.xlsx')
print(f'handled data {count}')