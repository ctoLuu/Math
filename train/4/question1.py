import math

C0 = 0.6
C1 = 0.3
t1 = 1.5
C_min = 0.05

k = -math.log(C1 / C0) / t1
t_min = -math.log(C_min / C0) / k


hours = int(t_min)
minutes = int((t_min - hours) * 60)

print(f"k:{k}")
print(f"t：{hours}小时{minutes}分钟")




