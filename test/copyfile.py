'''
@Project ：test 
@File    ：copyfile.py
@Author  ：yuk
@Date    ：2024/1/11 10:20 
description：将文件移动到另一文件夹
'''

import shutil
import os

def copy_file(src_path, dest_folder):
    try:
        # 检查目标文件夹是否存在，如果不存在则创建
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # 获取源文件名
        file_name = os.path.basename(src_path)

        # 构造目标路径
        dest_path = os.path.join(dest_folder, file_name)

        # 复制文件
        shutil.copy2(src_path, dest_path)

        print(f"文件 {file_name} 已成功复制到 {dest_folder}")
    except Exception as e:
        print(f"复制文件时发生错误: {e}")

# 示例使用
# source_file_path = r"Z:\liyunzhuo\detection\person_detection_data\real_data_1/000001.txt"
destination_txt_folder = "Z:\zjt\datasets\person\labels"
destination_img_folder = "Z:\zjt\datasets\person\images"
#
# copy_file(source_file_path, destination_folder)
root = r'Z:\liyunzhuo\detection\person_detection_data\widerperson2019_cleaned'

for file in os.listdir(root):
    if file.endswith('.txt'):
        source_file = os.path.join(root,file)
        copy_file(source_file, destination_txt_folder)
        # exit()
    elif file.endswith('.jpg'):
        source_file = os.path.join(root,file)
        copy_file(source_file, destination_img_folder)