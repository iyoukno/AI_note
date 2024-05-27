'''
@Project ：test 
@File    ：xml_to_voc.py
@Author  ：yuk
@Date    ：2024/1/19 15:46 
description：
'''
import os
import xml.etree.ElementTree as ET
classes = ["fire", "smoke"]

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    x = round(x,6)
    w = round(w,6)
    y = round(y,6)
    h = round(h,6)
    return x, y, w, h


def convert_annotation(image_id):
    try:
        # img_file = Image.open('D:\dataSet\pp_fall\images/%s.jpg' % (image_id))
        in_file = open(r'D:\datasets\smoke_fire\Smoke\Smoke\smoke\smoketrain/%s.xml' % (image_id), encoding='utf-8')
        out_file = open(r'D:\datasets\smoke_fire\Smoke\Smoke\smoke\labels/%s.txt' % (image_id), 'w', encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes and difficult != 0:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')
    except Exception as e:
        print(e, image_id)

# name = '1303901030777384962_1599984091_000'

path = r'D:\datasets\smoke_fire\Smoke\Smoke\smoke\smoketrain'

for file in os.listdir(path):
    file_name = file.split('.')[0]
    convert_annotation(file_name)