'''
@Project ：test 
@File    ：split.py
@Author  ：yuk
@Date    ：2024/1/11 11:09 
description：从文件夹中划分数据集
'''
import os
import random

root = r'Z:\zjt\datasets\fire_smoke\images'
list = os.listdir(root)

train_ratio = 0.8
val_ratio = 0.3
random.shuffle(list)

train_list = list
# val_list = list[int(train_ratio*len(list))-1:-1]
# train_list = random.sample(list,int(train_ratio*len(list)))
val_list = random.sample(list,int(val_ratio*len(list)))

train_label = open(r'Z:\zjt\datasets\fire_smoke\train.txt', 'w', encoding='utf8')
val_label = open(r'Z:\zjt\datasets\fire_smoke\val.txt', 'w', encoding='utf8')
for i in train_list:
    train_label.write(f'/data0/zjt/datasets/fire_smoke/images/{i}\n')

for j in val_list:
    val_label.write(f'/data0/zjt/datasets/fire_smoke/images/{j}\n')

train_label.close()
val_label.close()