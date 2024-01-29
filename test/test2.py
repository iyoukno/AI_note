'''
@Project ：test 
@File    ：test2.py
@Author  ：yuk
@Date    ：2023/11/30 9:35 
description：中文文档数据集映射标签
'''
import os

import cv2

# label_f = open('txt/label.txt', encoding='utf8')
# label = label_f.readlines()
# new_label = open('txt/data_test_label.txt', 'w', encoding='utf8')
# with open('txt/data_test.txt','r',encoding='utf8') as f:
#     lines = f.readlines()
#     for line in lines:
#         info = line.split(' ')
#         file_name = info[0]
#         file_label = info[1:]
#         label_txt = [label[int(i.strip('\n'))].strip('\n') for i in file_label]
#         str_info = ''
#         for i in label_txt:
#             str_info+=i
#         new_label.write(f"{file_name}\t{str_info}\n")
# label_f.close()
# new_label.close()

src_path = r'Z:\zjt\datasets\ocr\test1\images'
new_path = r'Z:\zjt\datasets\ocr\test1\images1'
with open('txt/data_test.txt','r',encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        info = line.split(' ')
        file_name = info[0]
        img = cv2.imread(os.path.join(src_path,file_name))
        cv2.imwrite(os.path.join(new_path,file_name),img)