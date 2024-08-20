import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import nonzero, array
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, rand_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei']

list = [[0, 1, 3, 6, 9, 10, 12, 15, 18, 21, 24, 27, 30, 31, 33, 34, 36, 39, 42, 45, 48, 51, 52, 53, 54, 55, 56, 57, 60, 63, 66, 69, 73, 76, 77, 79, 83, 84, 86, 87, 88, 89, 90, 91, 92, 95, 102, 103, 106, 109, 110, 112, 113, 115, 118, 121, 124, 127, 128, 130, 131, 133, 136, 139, 141, 142, 144, 145, 146, 148, 151, 154, 156, 157, 158, 160, 163, 164, 166, 167, 168, 169, 170, 172, 175, 178, 181, 184, 185, 187, 188, 189, 190, 191, 192, 193, 197, 198, 199, 200, 201, 202, 203, 204, 207, 210, 211, 212, 213, 214, 215, 216, 219, 222, 225, 228, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 243, 246, 249, 252, 253, 254, 255, 257, 260, 263, 264, 266, 267, 269, 270, 271, 272, 273, 274, 275, 278, 281, 284, 287, 290, 291, 292, 293, 294, 295, 296, 299, 302, 305, 308, 309, 311, 312, 313, 314, 315, 316, 317, 318, 320, 323, 326, 329, 332, 333, 334, 335, 336, 337, 338, 341, 344, 347, 348, 349, 350, 351, 352, 353, 354, 356, 357, 358, 359, 360, 361, 362, 363, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 415, 418, 420, 421, 423, 426, 431, 433, 436, 439, 440, 442, 445, 448, 451, 454, 457, 459, 460, 461, 463, 464, 469, 472, 477, 480, 483, 486, 487, 489, 492, 498, 501, 504, 505, 510, 513, 519, 525, 531, 534, 537, 540, 543, 546, 547, 549, 552, 558, 561, 564, 565, 566, 567, 570, 573, 579, 582, 585, 600, 605, 606, 609, 612, 620, 623, 626, 629, 632, 635, 638, 644, 650, 656, 659, 662, 665, 672, 673, 675, 681, 687, 690, 696, 699, 702, 718, 720, 723, 725, 728, 737, 741, 744, 747, 749, 750, 751, 752, 753, 755, 764, 773, 776, 779, 788, 791, 797, 800, 806, 807, 808, 809, 812, 815, 818, 825, 828, 830, 831, 833, 836, 848, 849, 851, 860, 863, 866, 869, 872, 875, 878, 881, 884, 887, 890, 891, 893, 896, 899, 902, 905, 911, 912, 914, 915, 917, 920, 923, 924, 925, 926, 927, 929, 930, 931, 932, 933, 935],
        [2, 4, 5, 7, 8, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 32, 35, 37, 38, 40, 41, 43, 44, 46, 47, 49,
         50, 58, 59, 61, 62, 64, 65, 67, 68, 70, 71, 72, 74, 75, 78, 80, 81, 82, 85, 93, 94, 96, 97, 98, 99, 100, 101,
         104, 105, 107, 108, 111, 114, 116, 117, 119, 120, 122, 123, 125, 126, 129, 132, 134, 135, 137, 138, 140, 143,
         147, 149, 150, 152, 153, 155, 159, 161, 162, 165, 171, 173, 174, 176, 177, 179, 180, 182, 183, 186, 194, 195,
         196, 205, 206, 208, 209, 217, 218, 220, 221, 223, 224, 226, 227, 229, 230, 241, 242, 244, 245, 247, 248, 250,
         251, 256, 258, 259, 261, 262, 265, 268, 276, 277, 279, 280, 282, 283, 285, 286, 288, 289, 297, 298, 300, 301,
         303, 304, 306, 307, 310, 319, 321, 322, 324, 325, 327, 328, 330, 331, 339, 340, 342, 343, 345, 346, 355, 364,
         413, 414, 417, 455, 456, 475, 482, 484, 491, 515, 528, 713, 724],
        [416, 428, 434, 435, 437, 438, 441, 443, 462, 465, 466, 467, 488, 493, 495, 506, 507, 516, 518, 521, 522, 526,
         529, 545, 548, 550, 553, 555, 559, 563, 568, 576, 586, 587, 588, 590, 591, 596, 601, 603, 607, 611, 614, 617,
         619, 621, 622, 625, 628, 630, 631, 633, 634, 636, 640, 641, 647, 651, 652, 653, 654, 657, 658, 660, 663, 669,
         674, 676, 677, 678, 679, 682, 684, 686, 688, 689, 693, 697, 698, 705, 708, 711, 714, 717, 719, 726, 727, 731,
         738, 739, 740, 742, 743, 745, 746, 748, 754, 758, 761, 767, 769, 770, 771, 772, 774, 775, 777, 778, 780, 782,
         786, 787, 789, 790, 792, 793, 794, 795, 798, 799, 801, 803, 804, 805, 810, 811, 814, 816, 821, 824, 826, 827,
         829, 832, 834, 835, 837, 839, 840, 842, 845, 846, 847, 850, 852, 853, 854, 857, 867, 868, 870, 871, 873, 888,
         889, 892, 908, 909, 910, 913, 916, 919, 921, 922, 928, 934],
        [419, 422, 424, 425, 427, 429, 430, 432, 444, 446, 447, 449, 450, 452, 453, 458, 468, 470, 471, 473, 474, 476, 478, 479, 481, 485, 490, 494, 496, 497, 499, 500, 502, 503, 508, 509, 511, 512, 514, 517, 520, 523, 524, 527, 530, 532, 533, 535, 536, 538, 539, 541, 542, 544, 551, 554, 556, 557, 560, 562, 569, 571, 572, 574, 575, 577, 578, 580, 581, 583, 584, 589, 592, 593, 594, 595, 597, 598, 599, 602, 604, 608, 610, 613, 615, 616, 618, 624, 627, 637, 639, 642, 643, 645, 646, 648, 649, 655, 661, 664, 666, 667, 668, 670, 671, 680, 683, 685, 691, 692, 694, 695, 700, 701, 703, 704, 706, 707, 709, 710, 712, 715, 716, 721, 722, 729, 730, 732, 733, 734, 735, 736, 756, 757, 759, 760, 762, 763, 765, 766, 768, 781, 783, 784, 785, 796, 802, 813, 817, 819, 820, 822, 823, 838, 841, 843, 844, 855, 856, 858, 859, 861, 862, 864, 865, 874, 876, 877, 879, 880, 882, 883, 885, 886, 894, 895, 897, 898, 900, 901, 903, 904, 906, 907, 918]
]

plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号的处理
plt.axes(aspect='equal')  # 将横、纵坐标轴标准化处理，确保饼图是一个正圆，否则为椭圆

length = 936
edu = [len(list[0]) / length, len(list[1]) / length, len(list[2]) / length, len(list[3]) / length]
labels = ['优', '良', '轻度污染', '重度污染']
explode = [0.1, 0, 0, 0]  # 生成数据，用于凸显大专学历人群
colors = ['#9999ff', '#ff9999', '#7777aa', '#2442aa']  # 自定义颜色

plt.pie(x=edu,  # 绘图数据
        explode=explode,  # 指定饼图某些部分的突出显示，即呈现爆炸式
        labels=labels,  # 添加教育水平标签
        colors=colors,
        autopct='%.2f%%',  # 设置百分比的格式，这里保留两位小数
        pctdistance=0.8,  # 设置百分比标签与圆心的距离
        labeldistance=1.1,  # 设置教育水平标签与圆心的距离
        startangle=180,  # 设置饼图的初始角度
        radius=1.2,  # 设置饼图的半径
        counterclock=False,  # 是否逆时针，这里设置为顺时针方向
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'green'},  # 设置饼图内外边界的属性值
        textprops={'fontsize': 10, 'color': 'black'},  # 设置文本标签的属性值
        )

# 添加图标题
# 显示图形
plt.show()