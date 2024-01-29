'''
@Project ：test 
@File    ：test.py
@Author  ：yuk
@Date    ：2023/11/29 16:07 
description：替换label中的空格
'''

n_l = open('new_label.txt','w',encoding='utf8')
with open('labels.txt','r',encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        if line.__contains__(' '):
            new_l = line.replace(' ','	')
        n_l.write(new_l)
n_l.close()