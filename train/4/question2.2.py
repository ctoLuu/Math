import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import f_oneway
plt.rcParams['font.sans-serif'] = ['SimHei']

# 数据
swimmers = np.array([0, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260])
cl_concen = np.array([0.4762, 0.4752, 0.4721, 0.4698, 0.4645, 0.4535, 0.4358, 0.4199, 0.4014, 0.3901, 0.3699, 0.3485, 0.3316, 0.3125, 0.3001])
swimmers_1 = np.array([0, 10, 20, 40])
cl_concen_1 = np.array([0.4762, 0.4752, 0.4721, 0.4698])
swimmers_2 = np.array([60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260])
cl_concen_2 = np.array([0.4645, 0.4535, 0.4358, 0.4199, 0.4014, 0.3901, 0.3699, 0.3485, 0.3316, 0.3125, 0.3001])
# 拟合函数
def exponential(x, a, b):
    return a * np.exp(b * x)

# 线性拟合
linear_params = np.polyfit(swimmers, cl_concen, 1)
linear_fit = np.poly1d(linear_params)

linear_params_1 = np.polyfit(swimmers_1, cl_concen_1, 1)
linear_fit_1 = np.poly1d(linear_params_1)

linear_params_2 = np.polyfit(swimmers_2, cl_concen_2, 1)
linear_fit_2 = np.poly1d(linear_params_2)

array = np.concatenate((linear_fit_1(swimmers_1), linear_fit_2(swimmers_2)), axis=0)
# 二次拟合
quadratic_params = np.polyfit(swimmers, cl_concen, 2)
quadratic_fit = np.poly1d(quadratic_params)

# 三次拟合
cubic_params = np.polyfit(swimmers, cl_concen, 3)
cubic_fit = np.poly1d(cubic_params)

# 指数拟合
exp_params, _ = curve_fit(exponential, swimmers, cl_concen, p0=(1, -0.01))
exp_fit = lambda x: exponential(x, *exp_params)

# 计算R2值
r2_linear = r2_score(cl_concen, linear_fit(swimmers))
r2_quadratic = r2_score(cl_concen, quadratic_fit(swimmers))
r2_cubic = r2_score(cl_concen, cubic_fit(swimmers))
r2_exp = r2_score(cl_concen, exp_fit(swimmers))
array = np.concatenate((linear_fit_1(swimmers_1), linear_fit_2(swimmers_2)), axis=0)
r2_linear_part = r2_score(cl_concen, array)
# 计算F值和p值
f_linear, p_linear = f_oneway(cl_concen, linear_fit(swimmers))
f_quadratic, p_quadratic = f_oneway(cl_concen, quadratic_fit(swimmers))
f_cubic, p_cubic = f_oneway(cl_concen, cubic_fit(swimmers))
f_exp, p_exp = f_oneway(cl_concen, exp_fit(swimmers))
f_linear_part, p_linear_part = f_oneway(cl_concen, array)
# 打印拟合函数、R2值、F值和p值
print(f"线性拟合函数: y = {linear_params[0]:.10f}x + {linear_params[1]:.10f}")
print(f"线性拟合 R2: {r2_linear:.10f}, F值: {f_linear:.10f}, p值: {p_linear:.10f}")

print(f"二次拟合函数: y = {quadratic_params[0]:.10f}x^2 + {quadratic_params[1]:.10f}x + {quadratic_params[2]:.10f}")
print(f"二次拟合 R2: {r2_quadratic:.10f}, F值: {f_quadratic:.10f}, p值: {p_quadratic:.10f}")

print(f"三次拟合函数: y = {cubic_params[0]:.10f}x^3 + {cubic_params[1]:.10f}x^2 + {cubic_params[2]:.10f}x + {cubic_params[3]:.10f}")
print(f"三次拟合 R2: {r2_cubic:.10f}, F值: {f_cubic:.10f}, p值: {p_cubic:.10f}")

print(f"指数拟合函数: y = {exp_params[0]:.10f} * e^({exp_params[1]:.10f}x)")
print(f"指数拟合 R2: {r2_exp:.10f}, F值: {f_exp:.10f}, p值: {p_exp:.10f}")

print(r2_linear_part)
print(f"F值: {f_linear_part:.10f}")
print(p_linear_part)
print(linear_fit_2)

# 绘图
plt.scatter(swimmers, cl_concen, label='数据点', color='black')
x = np.linspace(0, 260, 2600)
x1 = np.linspace(0, 40, 400)
x2 = np.linspace(40, 260, 2200)
# plt.plot(x, linear_fit(x), label='线性拟合', color='blue')
# plt.plot(x, quadratic_fit(x), label='二次拟合', color='green')
# plt.plot(x, cubic_fit(x), label='三次拟合', color='red')
# plt.plot(x, exp_fit(x), label='指数拟合', color='purple')
plt.plot(x, np.concatenate((linear_fit_1(x1), linear_fit_2(x2)), axis=0), label='指数拟合', color='blue')
plt.xlabel('游泳人数')
plt.ylabel('氯浓度（0.5小时）')
plt.legend()
plt.show()
