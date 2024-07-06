import numpy as np
import pandas as pd
import random

input_file_path = '结果表2-final.csv'
output_file_path = '结果表6.csv'

try:
    df = pd.read_csv(input_file_path, encoding='GB2312')
except UnicodeDecodeError:
    df = pd.read_csv(input_file_path, encoding='GBK')

df['日期'] = pd.to_datetime(df['日期'])

def assign_shift(hour):
    if 0 <= hour < 8:
        return '00:00-08:00'
    elif 5 <= hour < 13:
        return '05:00-13:00'
    elif 8 <= hour < 16:
        return '08:00-16:00'
    elif 12 <= hour < 20:
        return '12:00-20:00'
    elif 14 <= hour < 22:
        return '14:00-22:00'
    elif 16 <= hour < 24:
        return '16:00-24:00'
    return None

df['班次'] = df['小时'].apply(assign_shift)

daily_demands = df.pivot_table(index='日期', columns='班次', values='货量', aggfunc='sum').fillna(0)

all_shifts = ['00:00-08:00', '05:00-13:00', '08:00-16:00', '12:00-20:00', '14:00-22:00', '16:00-24:00']
for shift in all_shifts:
    if shift not in daily_demands.columns:
        daily_demands[shift] = 0

num_regular_employees = 200
max_temp_workers_per_shift = 300

result_data = []

# 初始化每日每员工的班次分配表
employee_shifts = {i: None for i in range(1, num_regular_employees + max_temp_workers_per_shift + 1)}

for shift in all_shifts:
    for date in daily_demands.index:
        demand = daily_demands.at[date, shift]
        assigned_regulars = 0
        assigned_temps = 0

        # 分配正式工
        for i in range(1, num_regular_employees + 1):
            if demand > 0 and employee_shifts[i] is None:
                employee_shifts[i] = shift
                result_data.append({
                    '分拣中心': 'SC60',
                    '日期': date.strftime('%Y-%m-%d'),
                    '班次': shift,
                    '出勤员工': f"正式工{i}"
                })
                assigned_regulars += 1
                demand -= 1

        # 分配临时工
        for i in range(num_regular_employees + 1, num_regular_employees + max_temp_workers_per_shift + 1):
            if demand > 0 and employee_shifts[i] is None:
                employee_shifts[i] = shift
                result_data.append({
                    '分拣中心': 'SC60',
                    '日期': date.strftime('%Y-%m-%d'),
                    '班次': shift,
                    '出勤员工': f"临时工{i - num_regular_employees}"
                })
                assigned_temps += 1
                demand -= 1

        # 重置分配，以便下一天重新使用员工
        for i in range(1, num_regular_employees + max_temp_workers_per_shift + 1):
            employee_shifts[i] = None

results_df = pd.DataFrame(result_data)
results_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f"Scheduling results saved to {output_file_path} with UTF-8-SIG encoding.")
