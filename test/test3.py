'''
@Project ：test 
@File    ：test3.py
@Author  ：yuk
@Date    ：2023/12/14 17:25 
description：改label中的path
'''

n_f = open('txt/ch_doc_label.txt','w', encoding='utf8')
with open('txt/data_test_label.txt','r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        n_f.write(f'images1/{line}')
n_f.close()
