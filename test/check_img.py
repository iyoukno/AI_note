'''
@Project ：test 
@File    ：check_img.py
@Author  ：yuk
@Date    ：2024/5/17 16:38 
description：检测img是否损坏，损坏的就删除
'''
import os

import cv2

root = r'Z:\zjt\yolov5-master\data\fire_smoke\negtive'

for i in os.listdir(root):
    try:
        img = cv2.imread(os.path.join(root,i))
        if img is None:
            os.remove(os.path.join(root, i))
            print(f'remove {i}')
    except:
        exit()