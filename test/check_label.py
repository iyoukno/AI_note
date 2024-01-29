'''
@Project ：test 
@File    ：check_label.py
@Author  ：yuk
@Date    ：2024/1/19 14:50 
description：检查yolo格式的标签是否正确
'''

import cv2

img_path = r'D:\datasets\smoke_fire\dataset\images\dir1_smoke_00350.jpg'
label_path = r'D:\datasets\smoke_fire\dataset\labels\dir1_smoke_00350.txt'

img = cv2.imread(img_path)
file = open(label_path,'r')



lines = file.readlines()

for line in lines:
    info = line.split(' ')
    cx, cy, w, h = [i.strip() for i in info[1:]]
    # 将归一化坐标恢复
    cx = float(cx) * img.shape[1]
    cy = float(cy) * img.shape[0]
    w = float(w) * img.shape[1]
    h = float(h) * img.shape[0]

    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()