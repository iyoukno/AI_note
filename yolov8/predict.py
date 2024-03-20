'''
@Project ：yolov8 
@File    ：predict.py
@Author  ：yuk
@Date    ：2024/3/15 14:22 
description：
'''
import cv2
from ultralytics import YOLO
from det_hand_meidiapipe import hand_detect

model = YOLO('runs/detect/train/weights/best.pt', task='detect')

result = model.predict([r'data/test2.jpg'], device='cpu')

# print(result[0].boxes.xyxy)
# res[0].show()


# for res in result:
#     # img_bgr = res.plot()
#     # 整张图传入
#     # hand_detect(img_bgr)
#     # 将检测到的roi传入
#     cx, cy, w, h = res.boxes.xywh.numpy().astype(int).tolist()[0]
#
#     # 将w,h 扩大
#     w, h = w * 2, h * 2.5
#     x1 = int(cx - w / 2) if int(cx - w / 2) > 0 else 0
#     y1 = int(cy - h / 2) if int(cy - h / 2) > 0 else 0
#     x2 = int(cx + w / 2) if int(cx + w / 2) > 0 else 0
#     y2 = int(cy + h / 2) if int(cy + h / 2) > 0 else 0
#     roi = res.orig_img[y1:y2, x1:x2]
#     annotated_roi_image, _ = hand_detect(roi)
#     res.orig_img[y1:y2, x1:x2] = annotated_roi_image
#
#     x1_o, y1_o, x2_o, y2_o = res.boxes.xyxy.numpy().astype(int).tolist()[0]
#     cv2.rectangle(res.orig_img, (x1_o, y1_o), (x2_o, y2_o), (0, 0, 255), 2)
#     cls = res.boxes.cls.numpy().astype(int)[0]
#     conf = res.boxes.conf.numpy()[0]
#     names = res.names[cls]
#     cv2.putText(res.orig_img, str(names)+str('%.2f' % conf),(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
#     # print(xyxy)
#     cv2.imshow('img', annotated_roi_image)
#     cv2.imshow('img2', res.orig_img)
#     cv2.waitKey(0)
#     # print(res.boxes.xyxy)
#     # res.show()
