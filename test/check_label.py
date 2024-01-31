'''
@Project ：test 
@File    ：check_label.py
@Author  ：yuk
@Date    ：2024/1/19 14:50 
description：检查yolo格式的标签是否正确
'''

import cv2

img_path = r'D:\datasets\FallDetection\images\000029.jpg'
label_path = r'D:\datasets\FallDetection\labels\000029.txt'

img = cv2.imread(img_path)
file = open(label_path,'r')



lines = file.readlines()
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
font_color = (0, 0, 255)  # 白色，格式为 (B, G, R)

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
    cv2.putText(img, 'fall' if info[0] == '0' else 'up', (int(x1), int(y1)), font, font_size, font_color, 1)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()