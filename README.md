# Math

## 数据预处理

### 缺失值

原来的部分-》对于小面积缺失值进行线性插值处理

增加-》对于大面积的缺失值进行线性相关分析，并根据计算的函数对缺失值进行补充，（以补充大面积缺失的NMHC(GT)为例）**建模补充 线性回归分析**

​	![image-20240708070100254](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20240708070100254.png)

figure 1：对C6H6与NMHC的线性回归分析

![image-20240708065244564](C:\Users\Asus\Desktop\pictures\image-20240708065244564.png)

figure 2：新增数据与线性回归分析

![image-20240708065419897](C:\Users\Asus\Desktop\pictures\image-20240708065419897.png)

### 异常值：

对于p值《=0.05（满足正态分布）使用3σ原则处理

对于p值》=0.05（不满足正太分布）使用箱型图处理

**建模补充 p值 正态分布**

figure ：数据的箱型图示意

![未命名绘图](C:\Users\Asus\Desktop\pictures\未命名绘图.png)

## 第一问：

figure 替换：显示为整型

![image-20240708071159895](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20240708071159895.png)
或者替换为：

![d2989dca61a35e8ea793527a18a072e](C:\Users\Asus\Desktop\pictures\d2989dca61a35e8ea793527a18a072e.png)

## 第二问

LSTM模型参数

- 单层LSTM单元数 10
- 全连接层神经元数 7
- 迭代次数 300
- 损失函数 均方误差
- 优化器 SGD
- 预测参考历史数据 24

figure 1：明显的日际变化规律示意图

![未命名绘图2](C:\Users\Asus\Desktop\pictures\未命名绘图2.png)

figure 2：使用lstm预测苯浓度的合理性

![未命名绘图3](C:\Users\Asus\Desktop\未命名绘图3.png)

figure 3：对苯浓度的未来一个月预测

![4c1e1af1110d40cf14be9a8f505dcbe](C:\Users\Asus\Desktop\pictures\4c1e1af1110d40cf14be9a8f505dcbe.png)

## 问题四

figure ：所有时间段各污染程度所占比重

![image-20240708075140154](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20240708075140154.png)

将每天的污染数据分为早中晚三个时段，对所有数据进行Kmeans聚类，按照聚类后的簇AQI均值将其环境质量分为优，良，轻度污染，中度污染，重度污染，严重污染。根据不同簇类污染物浓度大小推断出不同污染程度下污染物的特征和主要时间分布

表：不同簇类簇心数据大小示意图

| 环境质量 | AQI   | CO(GT) | NMHC(GT) | C6H6(GT) | NOx(GT) | NO2(GT) | Time |
| -------- | ----- | ------ | -------- | -------- | ------- | ------- | ---- |
| 优       | 32.92 | 0.95   | 48.05    | 4.56     | 89.95   | 62.13   | 早   |
| 良       | 49.03 | 1.71   | 115.75   | 8.31     | 136.17  | 94.63   | 早   |
| 轻度污染 | 57.63 | 1.30   | 46.78    | 4.99     | 210.58  | 109.65  | 中   |
| 中度污染 | 67.20 | 2.92   | 254.57   | 15.38    | 218.07  | 125.40  | 中   |
| 重度污染 | 82.03 | 2.22   | 124.61   | 9.40     | 347.70  | 154.91  | 晚   |
| 严重污染 | 92.71 | 3.48   | 260.72   | 16.94    | 467.69  | 161.06  | 晚   |

据表可知，NOx(GT),NO2(GT)与污染物程度具有强相关性，对污染程度影响最为严重，NMHC(GT)与C6H6(GT)对污染程度也有影响但影响相对较小，CO(GT)对污染程度影响较小。