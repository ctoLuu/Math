import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 读取地图和任务数据
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
        tasks.append(((sx, sy), (gx, gy)))
    print(tasks)
    return grid, tasks


# A*算法用于在网格中寻找最短路径
def a_star(grid, start, goal, constraints, agent_id):
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 四个方向：右、下、左、上

    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def in_bounds(pos):
        return 0 <= pos[0] < rows and 0 <= pos[1] < cols and grid[pos[0]][pos[1]] != '@'

    def is_constrained(pos, t):
        if (agent_id, pos, t) in constraints:
            return True
        if (agent_id, pos) in constraints:  # 永久禁止进入的点
            return True
        return False

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start), 0, start))
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            print(path[::-1])
            return path[::-1]

        for direction in directions:
            next_pos = (current[0] + direction[0], current[1] + direction[1])
            next_g = current_g + 1

            if in_bounds(next_pos) and not is_constrained(next_pos, next_g):
                if next_g < g_score[next_pos]:
                    g_score[next_pos] = next_g
                    priority = next_g + heuristic(next_pos)
                    heapq.heappush(open_set, (priority, next_g, next_pos))
                    came_from[next_pos] = current

        # 允许原地等待
        if not is_constrained(current, current_g + 1):
            if current_g + 1 < g_score[current]:
                g_score[current] = current_g + 1
                heapq.heappush(open_set, (current_g + 1 + heuristic(current), current_g + 1, current))
                came_from[current] = current

    return None


# CBS 高层算法
class CBSSolver:
    def __init__(self, grid, tasks):
        self.grid = grid
        self.tasks = tasks
        self.agents = len(tasks)
        self.node_id = 0

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
        for i, [start, goal] in enumerate(self.tasks):
            path = a_star(self.grid, start, goal, root['constraints'], i)
            if path is None:
                raise ValueError(f"无法找到机器人 {i + 1} 的无碰撞路径")
            root['paths'].append(path)
        root['cost'] = sum(len(path) - 1 for path in root['paths'])

        open_list = []
        heapq.heappush(open_list, (root['cost'], self.node_id, root))
        self.node_id += 1
        print(open_list)
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
                path = a_star(self.grid, self.tasks[agent][0], self.tasks[agent][1], child['constraints'], agent)
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
            if grid[r][c] == '@':
                ax.add_patch(patches.Rectangle((c, r), 1, 1, color='black'))
            else:
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
    grid, tasks = read_map_and_tasks('64x64map.txt')
    solver = CBSSolver(grid, tasks)
    paths = solver.solve()

    # 打印结果
    for i, path in enumerate(paths):
        print(f"机器人 {i + 1}:")
        print(f"路径: {path}")
        print(f"时间开销: {len(path) - 1}")

    # 可视化路径
    visualize(grid, paths)
