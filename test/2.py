'''
@Project ：test
@File    ：test2.py
@Author  ：yuk
@Date    ：2023/11/30 9:35
description：Msnet项目的训练数据制作，主要将标注的maks图片转成二值图
'''

import os

import cv2
import numpy as np

src_path = r'E:\facile_img_mask\outputs\attachments'
tar_path = r'E:\facile_img_mask\mask'


def convert(img_path):
    img = cv2.imread(img_path)
    idx = np.where(img!=0)
    idx_ = np.where(img == 0)
    img[idx] = 0
    img[idx_] = 255

    # _,_,_,mask = cv2.split(img)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return img


for i in os.listdir(src_path):
    img_path = os.path.join(src_path, i)
    img = convert(img_path)
    cv2.imwrite(f"{tar_path}/{i[:-6]}.png", img)
    # exit()

# img = cv2.imread(r'img/1.png',0)
# cv2.imshow('img', img)
# cv2.waitKey(0)