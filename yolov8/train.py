'''
@Project ：yolov8 
@File    ：train.py
@Author  ：yuk
@Date    ：2024/3/13 16:18 
description：
'''
from ultralytics import YOLO

model = YOLO('yolov8s.pt', task='detect')

re = model.train(data='cfg/phone.yaml', epochs=100, imgsz=640, batch=24, device=0)