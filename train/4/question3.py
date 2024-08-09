import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
plt.rcParams['font.sans-serif'] = ['SimHei']

# 初始化参数
C0 = 0.6
k_pause = 0.462098
opera_thr = 0.3
pause_thr = 0.05
cl_value = 0.6

# 定义营业时间段和人数
time_slots = [("9:00", "10:00"), ("10:00", "11:00"), ("11:00", "12:00"),
              ("12:00", "13:00"), ("13:00", "14:00"), ("14:00", "15:00"),
              ("15:00", "16:00"), ("16:00", "17:00"), ("17:00", "18:00"),
              ("18:00", "19:00"), ("19:00", "20:00"), ("20:00", "21:00")]

swimmers = [20, 80, 0, 150, 180, 0, 210, 340, 0, 280, 420, 190]

# 存储浓度变化和加氯时间
times = []
concen = []
cl_times = []

# 存储每个时间段的0.5小时后的浓度和k值
half_hour_concen = []
decay_rates = []

current_time = datetime.strptime("9:00", "%H:%M")
current_concen= C0

def v2(N):
    return 0.4812537698 + -0.0003123067 * N - 0.0000016057 * N ** 2

# 开始计算每分钟的浓度变化
for i, (start_time, end_time) in enumerate(time_slots):
    slot_start = datetime.strptime(start_time, "%H:%M")
    slot_end = datetime.strptime(end_time, "%H:%M")
    N = swimmers[i]

    # 获取当前时间段开始时的k值
    if N > 0:
        C1 = v2(N)
        k = -math.log(C1 / C0) / 0.5
    else:
        k = k_pause

    half_hour_concen.append(C1)
    decay_rates.append(k)

    while current_time < slot_end:
        times.append(current_time)
        concen.append(current_concen)

        # 更新浓度
        current_concen *= np.exp(-k * 1 / 60)

        # 检查是否需要加氯
        if (N > 0 and current_concen < opera_thr) or (
                N == 0 and current_concen < pause_thr):
            cl_times.append(current_time)
            current_concen = cl_value

        current_time += timedelta(minutes=1)

# 自定义刻度标签
hour_labels = [f"{hour}:00" for hour in range(9, 22)]

# 绘制浓度变化曲线
plt.figure(figsize=(12, 6))
plt.plot(times, concen, label='余氯浓度')

# 标记加氯时间
for cl_time in cl_times:
    plt.axvline(x=cl_time, color='r', linestyle='--',
                label='加氯' if cl_time == cl_times[0] else "")

# 标记阈值线
plt.axhline(opera_thr, color='y', linestyle='--', label=f'运营加氯阈值 {opera_thr} mg/L')
plt.axhline(pause_thr, color='g', linestyle='--', label=f'维护加氯阈值 {pause_thr} mg/L')
plt.axhline(0.6, color='b', linestyle='--', label='初始浓度 0.6 mg/L')

# 添加运营和闭馆时间段的背景颜色
plt.axvspan(datetime.strptime("11:00", "%H:%M"), datetime.strptime("12:00", "%H:%M"), color='#72F2EB', alpha=0.5)
plt.axvspan(datetime.strptime("14:00", "%H:%M"), datetime.strptime("15:00", "%H:%M"), color='#72F2EB', alpha=0.5)
plt.axvspan(datetime.strptime("17:00", "%H:%M"), datetime.strptime("18:00", "%H:%M"), color='#72F2EB', alpha=0.5)

plt.xlabel('时间')
plt.ylabel('余氯浓度 (mg/L)')
plt.title('9:00-21:00 余氯浓度变化曲线')
plt.xticks([datetime.strptime(f"{hour}:00", "%H:%M") for hour in range(9, 22)], labels=hour_labels)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 输出加氯时间
for ct in cl_times:
    print(f"加氯时刻：{ct.strftime('%H:%M')}")

# 输出每个运营时间段0.5小时后的浓度及k值
for i, (start_time, end_time) in enumerate(time_slots):
    print(f"{start_time}-{end_time} 时间段的 k 值: {decay_rates[i]:.6f}, 0.5小时后的余氯浓度: {half_hour_concen[i]:.6f} mg/L")
