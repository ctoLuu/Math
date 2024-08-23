from random import randint, seed
import copy
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

seed(20000)


def read_map_and_tasks(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    # 读取地图大
    rows, cols = map(int, lines[1].split())
    grid = []
    # 读取地图
    for i in range(2, rows + 2):
        grid.append(lines[i].split())
    # 读取任务信息
    tasks = []
    task_start_index = rows + 3
    num_tasks = int(lines[task_start_index])
    for i in range(task_start_index + 1, task_start_index + 1 + num_tasks):
        sx, sy, gx, gy = map(int, lines[i].split())
        tasks.append(((sy, sx), (gy, gx)))
    print(tasks)
    return grid, tasks


class SearchEntry():

    def __init__(self, x, y, g_cost, f_cost=0, pre_entry=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.pre_entry = pre_entry

    def getPos(self):
        return (self.x, self.y)

    def __str__(self):
        return 'x = {}, y = {}, f = {}'.format(str(self.x), str(self.y), str(self.f_cost))


class Map():

    def __init__(self, width, height, grid):
        self.width = width
        self.height = height
        self.grid = grid
        self.map = [[0 for x in range(self.width)] for y in range(self.height)]
        self.createMap(grid)

    def createMap(self, filename):
        for y in range(self.height):
            for x in range(self.width):
                if grid[y][x] == '@':
                    self.map[y][x] = 1
                elif grid[y][x] == '.':
                    self.map[y][x] = 0

    def showMap(self):
        print("+" * (3 * self.width + 2))

        for row in self.map:
            s = '+'
            for entry in row:
                s += ' ' + str(entry) + ' '
            s += '+'
            print(s)

        print("+" * (3 * self.width + 2))


def AStarSearch(map, source, dest):

    def getNewPosition(map, locatioin, offset):
        x, y = (location.x + offset[0], location.y + offset[1])
        if x < 0 or x >= map.width or y < 0 or y >= map.height or map.map[y][x] == 1:
            return None
        return (x, y)

    def getPositions(map, location):
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        poslist = []
        for offset in offsets:
            pos = getNewPosition(map, location, offset)
            if pos is not None:
                poslist.append(pos)
        return poslist

    def calHeuristic(pos, dest):
        return abs(dest.x - pos[0]) + abs(dest.y - pos[1])

    def getMoveCost(location, pos):
        if location.x != pos[0] and location.y != pos[1]:
            return 1.4
        else:
            return 1

    def isInList(list, pos):
        if pos in list:
            return list[pos]
        return None

    def addAdjacentPositions(map, location, dest, openlist, closedlist):
        poslist = getPositions(map, location)
        for pos in poslist:
            if isInList(closedlist, pos) is not None:
                continue

            findEntry = isInList(openlist, pos)
            h_cost = calHeuristic(pos, dest)
            g_cost = location.g_cost + getMoveCost(location, pos)
            if findEntry is None:
                openlist[pos] = SearchEntry(pos[0], pos[1], g_cost, g_cost + h_cost, location)
            elif findEntry.g_cost > g_cost:
                findEntry.g_cost = g_cost
                findEntry.f_cost = g_cost + h_cost
                findEntry.pre_entry = location

    def getFastPosition(openlist):
        fast = None
        for entry in openlist.values():
            if fast is None:
                fast = entry
            elif fast.f_cost > entry.f_cost:
                fast = entry
        return fast

    path = []

    openlist, closedlist = {}, {}
    location = SearchEntry(source[0], source[1], 0.0)
    dest = SearchEntry(dest[0], dest[1], 0.0)
    openlist[source] = location

    while True:
        location = getFastPosition(openlist)
        if location is None:
            print("can't find valid path")
            break

        if location.x == dest.x and location.y == dest.y:
            break

        closedlist[location.getPos()] = location
        openlist.pop(location.getPos())
        addAdjacentPositions(map, location, dest, openlist, closedlist)

    while location is not None:
        print(location)
        path.append((location.x, location.y))
        map.map[location.y][location.x] = 2
        location = location.pre_entry
    path.reverse()
    print(path)
    return path

# CBS 高层算法
class CBSSolver:
    def __init__(self, grid, tasks, map):
        self.grid = grid
        self.tasks = tasks
        self.agents = len(tasks)
        self.map = map
        self.node_id = 0  # 用于生成唯一的标识符

    def detect_conflict(self, paths):
        max_t = max(len(path) for path in paths)
        for t in range(max_t):
            occupied = {}
            for i, path in enumerate(paths):
                pos = path[t] if t < len(path) else path[-1]
                if pos in occupied:
                    return (occupied[pos], i, pos, t)
                occupied[pos] = i

            for i in range(self.agents):
                if t >= len(paths[i]):
                    continue
                for j in range(i + 1, self.agents):
                    if t >= len(paths[j]):
                        continue
                    if paths[i][t] == paths[j][t] and paths[i][t - 1] == paths[j][t - 1]:
                        return (i, j, paths[i][t], t)
        return None

    def solve(self):
        root = {
            'constraints': [],
            'paths': [],
            'cost': 0
        }

        for i, [start, goal] in enumerate(tasks):
            copy_map = copy.deepcopy(map)
            print(start, goal)
            path = AStarSearch(copy_map, start, goal)
            root['paths'].append(path)
            copy_map.showMap()
        root['cost'] = sum(len(path) - 1 for path in root['paths'])

        open_list = []
        heapq.heappush(open_list, (root['cost'], self.node_id, root))
        self.node_id += 1

        while open_list:
            _, _, node = heapq.heappop(open_list)
            conflict = self.detect_conflict(node['paths'])
            if conflict is None:
                return node['paths']

            i, j, pos, t = conflict
            for agent, position in [(i, pos), (j, pos)]:
                child = {
                    'constraints': node['constraints'] + [(agent, position, t)],
                    'paths': node['paths'][:],
                    'cost': 0
                }
                path = AStarSearch(self.map, self.tasks[agent][0], self.tasks[agent][1])
                if path is not None:
                    child['paths'][agent] = path
                    child['cost'] = sum(len(p) - 1 for p in child['paths'])
                    heapq.heappush(open_list, (child['cost'], self.node_id, child))
                    self.node_id += 1

        return None

# 可视化函数
def visualize(grid, paths):
    fig, ax = plt.subplots()
    rows, cols = len(grid), len(grid[0])

    # 绘制网格和障碍物
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                ax.add_patch(patches.Rectangle((c, r), 1, 1, color='black'))
            elif grid[r][c] == '0':
                ax.add_patch(patches.Rectangle((c, r), 1, 1, edgecolor='gray', facecolor='white'))

    # 绘制路径
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    for i, path in enumerate(paths):
        color = colors[i % len(colors)]
        for t in range(len(path)):
            # 将路径的坐标进行翻转
            x, y = path[t]
            flipped_x, flipped_y = y, x  # 翻转坐标

            if t > 0:
                prev_x, prev_y = path[t - 1]
                flipped_prev_x, flipped_prev_y = prev_y, prev_x  # 翻转之前的坐标
                ax.plot(
                    [flipped_prev_x + 0.5, flipped_x + 0.5],
                    [flipped_prev_y + 0.5, flipped_y + 0.5],
                    color=color, linewidth=2
                )

        # 绘制起点和终点
        start_x, start_y = path[0][1], path[0][0]  # 翻转坐标
        goal_x, goal_y = path[-1][1], path[-1][0]  # 翻转坐标
        ax.plot(start_x + 0.5, start_y + 0.5, 'o', color=color, markersize=8, label=f'Agent {i + 1} Start')  # 起点标记
        ax.plot(goal_x + 0.5, goal_y + 0.5, 'x', color=color, markersize=8)  # 终点标记

        # 标注起点和终点的编号
        ax.text(start_x + 0.5, start_y + 0.5, f'{i + 1}', ha='center', va='center', fontsize=10, color='white')
        ax.text(goal_x + 0.5, goal_y + 0.5, f'{i + 1}', ha='center', va='center', fontsize=10, color='white')

    # 添加图例
    legend_handles = [
        patches.Patch(color=colors[i % len(colors)], label=f'Agent {i + 1}')
        for i in range(len(paths))
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))

    # 设置网格和比例
    ax.set_xticks([x for x in range(cols)])
    ax.set_yticks([y for y in range(rows)])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    WIDTH = 8
    HEIGHT = 8
    file = './8x8map.txt'
    grid, tasks = read_map_and_tasks(file)
    map = Map(WIDTH, HEIGHT, grid)

    solver = CBSSolver(grid, tasks, map)
    paths = solver.solve()