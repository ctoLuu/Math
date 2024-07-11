import numpy as np
import pandas as pd


def comp_fit(matrix,array):
    time = 0
    for i in range(1, 10):
        for j in range(10):
            current_i = i
            while (matrix[current_i - 1, j] == 0) & (current_i != -1):
                current_i -= 1
            if current_i != -1:
                time += array[matrix[current_i - 1, j] - 1, matrix[i, j] - 1]
    # print(time)
    return time
a = np.array([[ 0, 0, 0,29,23, 5,25, 8,28,22],
 [ 0  ,0  ,0  ,6,26 ,0 ,15, 9,12,23],
 [ 0 , 0, 22 ,29, 17 , 0, 27 ,15 ,11 , 7],
 [ 0 , 0  ,0  ,0 ,17 , 0 , 0 , 0 , 0 , 8],
 [17  ,0 ,11  ,6, 22 ,29 , 0, 15  ,0, 26],
 [24 , 0 , 0  ,0 , 0 , 9 , 8,  0, 21 , 0],
 [26  ,0 , 5 ,23 , 0 , 9 , 3 ,24 ,15  ,4],
 [20 ,29  ,0 ,26 ,22, 17 ,18 , 4  ,2, 13],
 [ 7  ,6 , 0  ,0, 23, 25, 28, 22 , 0, 24],
 [ 0  ,0 , 0  ,0  ,0, 19, 23 , 0  ,0 , 0]])
print(a)
df1 = pd.read_excel('./附件3/Ins2_10_30_10.xlsx', sheet_name=1)
lists = []
df1['所需墨盒编号'] = df1['所需墨盒编号'].apply(lambda x: eval(x))
for index, row in df1.iterrows():
    lists.append(row['所需墨盒编号'])

df2 = pd.read_excel('./附件3/Ins2_10_30_10.xlsx', sheet_name=2)
df2.drop(columns=['Unnamed: 0'], inplace=True)
array = df2.values
array = np.array(array)
slot_num = 10
print(comp_fit(a,array))
