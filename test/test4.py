'''
@Project ：test 
@File    ：test4.py
@Author  ：yuk
@Date    ：2023/12/14 17:31 
description：合并字符字典
'''

keys1 = open('txt/idcard_keys_.txt', 'r', encoding='utf8')
keys2 = open('txt/char_std_5990.txt', 'r', encoding='utf8')

with open('txt/idcard_uni_keys.txt', 'w', encoding='utf8') as f:
    k1_lines = keys1.readlines()
    k2_lines = keys2.readlines()
    key = set().union(k1_lines,k2_lines)
    # print(k1_lines)
    f.writelines(key)



keys1.close()
keys2.close()