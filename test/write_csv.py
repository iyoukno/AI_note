'''
@Project ：test 
@File    ：write_csv.py
@Author  ：yuk
@Date    ：2024/5/16 11:02 
description：将txt的数据写入csv
'''
import csv
# field_names =
txt = [['image_name','标签中火焰数量','标签中烟雾数量','预测的火焰数量','预测的烟雾数量']]

fire_label_P = 0
fire_pred_P = 0
smoke_label_P = 0
smoke_pred_P = 0

with open('res.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        # txt.append(line.rstrip('\n').split('\t'))
        data = line.rstrip('\n').split('\t')
        # label中火焰的样本数量（包含label中没标记的，但模型检测到了的）
        if (data[1] == '0' and data[3] !='0') or (data[1] != '0'):
            fire_label_P += 1
        # 检测到的火焰的样本数量
        if data[3] !='0':
            fire_pred_P += 1
        # label中烟雾的样本数量（包含label中没标记的，但模型检测到了的）
        if (data[2] == '0' and data[4] !='0') or (data[2] != '0'):
            smoke_label_P += 1
        # 检测到的烟雾的样本数量
        if data[4] !='0':
            smoke_pred_P += 1

    # print(lines)
    print(f'fire_label_P：{fire_label_P}\tfire_pred_P：{fire_pred_P}\nsmoke_label_P：{smoke_label_P}\tsmoke_pred_P：{smoke_pred_P}')

# 写入csv
# with open('example.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerows(txt)