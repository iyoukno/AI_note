'''
@Project ：test 
@File    ：seg_img_openv.py
@Author  ：yuk
@Date    ：2025/2/21 11:50 
description：使用opencv将目标图片分割出来，然后贴到背景上，用作数据增强
'''
import random

import cv2
import numpy as np
from PIL import Image

target_img = cv2.imread('../img/idcard/12.jpg')

bg_img = cv2.imread('../img/idcard/bg/IMG_0073.JPG')

#等比例缩放
h ,w = target_img.shape[0:2]
ratio = h/w
# random_size = [200,250,200,250]
random_size = [600,650,700,880]
new_h = random.choice(random_size)#random.randint(400,600)
new_shape = (int(new_h/ratio),new_h)
# opencv的resize 的shape是：w,h   w必须在前面
target_img = cv2.resize(target_img,new_shape)

target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img',target_img_gray)
# cv2.waitKey(0)

# 找轮廓抠图
retval, logoBinary = cv2.threshold(target_img_gray,120,255,cv2.THRESH_OTSU)
cv2.imshow('logoBinary', logoBinary)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(logoBinary,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
contours_list = list(contours)
contours_list_sorted = sorted(contours_list, key=lambda x:x.shape[0],reverse=True)
mask = np.zeros_like(target_img,dtype='u1')
cv2.drawContours(mask,contours_list_sorted,0,color=(255, 255, 255),thickness=-1)
cv2.imshow('mask',mask)
cv2.waitKey(0)

idx = np.where(mask == 255)
#单次测试
# Y_mark = random.randint(0,(bg_img.shape[0]-new_h)); X_mark = random.randint(0,(bg_img.shape[1]-new_shape[0]))
# copyimg = bg_img[Y_mark:Y_mark + new_h, X_mark:X_mark + new_shape[0]].copy()
# copyimg[idx] = target_img[idx]
# bg_img[Y_mark:Y_mark + new_h, X_mark:X_mark + new_shape[0]] = copyimg
#
# cv2.imshow('new_img',bg_img)
# cv2.waitKey(0)
# cv2.imwrite('res.jpg',bg_img)


for i in range(5):
    bg_img_temp = bg_img.copy()
    # 随机产生起始坐标点
    Y_mark = random.randint(0,(bg_img.shape[0]-new_h-10)); X_mark = random.randint(0,(bg_img.shape[1]-new_shape[0]-10))
    copyimg = bg_img_temp[Y_mark:Y_mark + new_h, X_mark:X_mark + new_shape[0]].copy()
    copyimg[idx] = target_img[idx]
    bg_img_temp[Y_mark:Y_mark + new_h, X_mark:X_mark + new_shape[0]] = copyimg

    # cv2.imshow('new_img',bg_img_temp)
    # cv2.waitKey(0)
    image = Image.fromarray(cv2.cvtColor(bg_img_temp, cv2.COLOR_BGR2RGB))
    # 随机角度
    angle = random.randint(-5,5)
    img = image.rotate(angle)
    img.save(f'../runs/img12_IMG_0073_{i}.jpg')
