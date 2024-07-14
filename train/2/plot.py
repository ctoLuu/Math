import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AN:
    def draw_paths(self, all_paths, CityCoordinates):
        plt.figure(figsize=(10, 8), dpi=150)  # 设置图像尺寸和分辨率
        for index, route in enumerate(all_paths):
            x, y = [], []
            for city_index in route:
                coordinate = CityCoordinates[city_index]
                x.append(coordinate[0])
                y.append(coordinate[1])

            # 绘制路径，并标记起点和终点
            plt.plot(x, y, marker='o', markersize=4, alpha=0.8, linewidth=1, label=f'car {index + 1}')
            plt.scatter([x[0], x[-1]], [y[0], y[-1]], s=60, edgecolor='black')  # 起点和终点加粗

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('GA_car_path')
        plt.legend()
        plt.grid(True)
        plt.show()

    def main(self):
        # 读取距离矩阵
        time_matrix_file = 'time_matrix.xlsx'
        dist_matrix_df = pd.read_excel(time_matrix_file, index_col=0)
        dist_matrix = dist_matrix_df.fillna(dist_matrix_df.values.max()).values

        # 读取坐标数据，并预处理
        data_file = '数据.xlsx'
        coordinates_df = pd.read_excel(data_file)
        coordinates_df = coordinates_df.drop(index=[2, 6, 22])  # 删除指定的异常数据行
        coordinates_df = coordinates_df.reset_index(drop=True)  # 重置索引以避免后续处理错误

        # 添加出发点坐标
        start_point = pd.DataFrame({'经度': [120.453], '纬度': [31.50]}, index=[0])
        coordinates_df = pd.concat([start_point, coordinates_df], ignore_index=True)
        CityCoordinates = coordinates_df[['经度', '纬度']].values.tolist()

        # 定义车辆路径
        all_paths = [
            [0,  5, 37, 11, 14,  0],
            [0 ,30, 29, 39, 33, 13, 19 ,31 , 0],
            [0 ,21 ,43 ,22 ,32 ,34 , 0],
            [0 , 4 , 3 ,42 ,20 ,10 , 2 ,38,  0],
            [00 ,27 ,25 ,23 ,26 ,24  ,8  ,6  ,9  ,7 ,18,  0],
            [0, 17, 15,  1,  0],
            [0, 35, 40, 36, 12, 41, 28, 16]
        ]

        self.draw_paths(all_paths, CityCoordinates)

if __name__ == "__main__":
    an = AN()
    an.main()
