'''
@Project ：yolov8 
@File    ：interface.py
@Author  ：yuk
@Date    ：2024/3/19 10:16 
description：
'''
import cv2
import torch

from utils.predict import DetectPhoneAndHand
model_path = r'/data0/zjt/yolov8/expo/det_phone.onnx'
img = cv2.imread(r'/data0/zjt/yolov8/data/vlcsnap-2024-03-19-09h59m50s994.png')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = DetectPhoneAndHand(model_path, device=device)
# xyxy:是手机框的坐标；annotated_image：是已经画好了手爪爪的ndarray，后续像画框框就在这上面画；hand_landmarks：手的坐标
xyxy, annotated_image, hand_landmarks= d.run(img)

print(xyxy)
print(hand_landmarks)