import pandas as pd  # 导入pandas库，用于数据处理

# 读取Excel文件并获取第一个工作表的数据
df1 = pd.read_excel('./附件1/Ins5_30_60_10.xlsx', sheet_name=1)

# 初始化一个空列表，用于存储每一行的墨盒编号列表
lists = []

# 使用apply函数和eval将'所需墨盒编号'列中的字符串转换为列表
df1['所需墨盒编号'] = df1['所需墨盒编号'].apply(lambda x: eval(x))
for index, row in df1.iterrows():  # 遍历DataFrame中的每一行
    lists.append(row['所需墨盒编号'])  # 将每行的墨盒编号添加到lists列表中

# 初始化变量
slot_count = 10   # 插槽数量
slot_number = []    # 用于存储当前插槽内的墨盒编号
leave_position = 0   # 记录第一次包装后剩余的插槽数量
change_count = 0    # 记录更换墨盒的次数
last_slot = []

# 判断第一次放置墨盒时是否需要留下一些插槽空着
if len(lists[0]) < slot_count:
    leave_position = slot_count - len(lists[0])  # 计算剩余插槽数量
    for number in lists[0]:  # 将第一次的墨盒编号放入插槽
        slot_number.append(number)
else:
    for number in lists[0]:
        slot_number.append(number)

# 遍历lists中的每一项，进行墨盒的比较和更换
for index, list in enumerate(lists):
    different = []  # 存放与插槽内不同的墨盒编号
    equal = []  # 存放与插槽内相同的墨盒编号
    different_num = 0  # 不同墨盒编号的数量
    equal_num = 0  # 相同墨盒编号的数量

    if index == 0:
        print(slot_number)
        continue  # 跳过第一次的遍历，因为第一次没有更换

    # 比较当前包装与插槽内的墨盒，分类存放在different和equal列表中
    for number in list:
        if number in slot_number:
            equal.append(number)
            equal_num += 1
        else:
            different.append(number)
            different_num += 1

    leave = leave_position
    # 如果有剩余插槽，先填充剩余插槽
    while (leave_position != 0) & (len(different) != 0):
        x = different.pop()  # 从不同的墨盒编号中取出一个
        slot_number.append(x)  # 添加到插槽编号列表中
        leave_position -= 1  # 减少剩余插槽数量
        different_num -= 1  # 减少不同墨盒编号的数量



    # 标记当前正在检查的包装索引
    flag = index + 1

    # 如果存在需要更换的墨盒，确定需要更换的插槽
    if len(different) != 0:
        while(different_num < len(slot_number) - leave - equal_num):
            if flag == len(lists):  # 如果已经检查到最后一个包装，跳出循环
                break
            # 检查后面的包装，确定被更换的插槽
            for number in lists[flag]:
                if (number in slot_number) & (number not in equal):
                    equal.append(number)
                    equal_num += 1
                elif number in equal:
                    continue
                elif number not in different:
                    different_num += 1
            flag += 1

    # 如果存在需要更换的墨盒，执行更换操作并更新计数器
    if len(different) != 0:
        for slot_index, number in enumerate(slot_number):
            if number in equal:
                continue
            else:
                change_count += 1  # 增加更换次数计数器
                x = different.pop()  # 取出一个需要更换的墨盒编号
                slot_number[slot_index] = x  # 更新插槽编号列表
                if len(different) == 0:
                    break
    print(slot_number)  # 打印当前插槽内的墨盒编号

# 打印总的更换次数
print(change_count)