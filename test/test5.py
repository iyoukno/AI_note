'''
@Project ：test 
@File    ：test5.py
@Author  ：yuk
@Date    ：2023/12/26 17:19 
description：文件rename
'''
import os
root = r'D:\datasets\VOC2012\JPEGImages'

for i in os.listdir(root):
    new_name = 'dir1_'+i
    os.rename(os.path.join(root,i),os.path.join(root,new_name))
    # print(i)
    # exit()