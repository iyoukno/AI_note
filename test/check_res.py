'''
@Project ：test 
@File    ：check_res.py
@Author  ：yuk
@Date    ：2024/5/16 10:14 
description：
'''
import os

res_pth = r'Z:\zjt\yolov5-master\runs\detect\exp4\labels'
label_pth = r'Z:\zjt\datasets\fire_smoke\labels'
out = open('res.txt', 'w')



for i in os.listdir(res_pth):
    real_fire = 0
    real_s = 0
    pred_fire = 0
    pred_s = 0
    print(i)
    with open(os.path.join(label_pth, i), 'r') as label_f:
        label_lines = label_f.readlines()
        for line in label_lines:
            if line[0] == '0':
                real_fire += 1
            if line[0] == '1':
                real_s += 1
            # print(line)


    with open(os.path.join(res_pth, i), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '0':
                pred_fire += 1
            if line[0] == '1':
                pred_s += 1
            # print(line)

    str_info = f'{i[:-4]}.jpg\t{real_fire}\t{real_s}\t{pred_fire}\t{pred_s}\n'
    out.write(str_info)
out.close()