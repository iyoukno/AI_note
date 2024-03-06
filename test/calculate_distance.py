'''
@Project ：test 
@File    ：calculate_distance.py
@Author  ：yuk
@Date    ：2024/3/6 15:07 
description：The straight-line distance of the point to the line
'''
import math

point = (50, 50)
# 若物体是逆着坐标轴方向移动，则line中起始点始终是y坐标小的为第一个点，反之第二,（特殊的，划线平行与x轴的，逆行：x大的为第一个点，反之第二）
line = [100,100, 0, 100]
# 若物体是顺着坐标轴移动
# line = []
A = line[3] - line[1]
B = line[0] - line[2]
C = line[2] * line[1] - line[0] * line[3]
# d = |A*point[0] + B*point[1] + C
d = (A * point[0] + B * point[1] + C) / math.sqrt(math.pow(A,2) + math.pow(B,2))

print(d)