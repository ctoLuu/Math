import numpy as np
def has_alternate_pattern_np(arr):
    # 检查数组长度是否至少为3
    if arr.size < 3:
        return False

    # 检查"0, 1, 0"模式
    pattern_010 = (arr[:-2] == 0) & (arr[1:-1] == 1) & (arr[2:] == 0)
    # 检查"1, 0, 1"模式
    pattern_101 = (arr[:-2] == 1) & (arr[1:-1] == 0) & (arr[2:] == 1)

    # 任何模式的组合存在至少一次即为True
    return np.any(pattern_010) or np.any(pattern_101)

a = np.array([0, 1, 0, 1, 0, 1, 2, 0, 4, 5, 6])
print(has_alternate_pattern_np(a))
print(np.sum(a==0))
zero_indices = np.where(a == 0)[0]
print(zero_indices)
sub_arrays = []
start_index = 0
for index in zero_indices:
    sub_arrays.append(a[start_index:index])
    start_index = index + 1
sub_arrays.append(a[start_index:])
print(sub_arrays)