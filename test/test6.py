'''
@Project ：test 
@File    ：test6.py
@Author  ：yuk
@Date    ：2024/4/24 15:30 
description：将一个文件夹里的图片resize, 针对yolo里的分类，图片预处理，作者采用中心裁切并resize的方法，导致数据不完整
yolov5 分类训练，参考：https://blog.csdn.net/small_wu/article/details/127239592
'''
import os

import cv2
size = (80,80)
root = r'Z:\zjt\datasets\hand_class\train\0'

for i in os.listdir(root):
    img_path = os.path.join(root,i)
    img = cv2.imread(img_path)
    r_img = cv2.resize(img,size)
    # cv2.imshow('img', r_img)
    # cv2.waitKey(0)
    # print(r_img)
    cv2.imwrite(img_path, r_img)
    print(img_path)
