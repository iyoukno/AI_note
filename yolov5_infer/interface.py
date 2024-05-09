'''
@Project ：detect fire and smoke
@File    ：det_onnx.py
@Author  ：yuk
@Date    ：2024/3/14 10:35 
description：
'''

import sys

import cv2

# sys.path.append(r'/data0/zjt/v5_infer_onnx2/pac')
sys.path.append(r'D:\project\yolov5_infer\pac')


from pac.det.det_onnx import *
from pac.classify.predict import *

onnx_ph = r'D:\project\yolov5_infer\pac\det\models\hand.onnx'
img_ph = r'D:\project\yolov5_infer\pac\det\data\0020.jpg'
class_model_ph = r'D:\project\yolov5_infer\pac\classify\models\best.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
im0 = cv2.imread(img_ph)
if __name__ == '__main__':
    d = Detect_by_onnx(onnx_path=onnx_ph, devicestr=device)
    c = Classfier(class_model_ph,device=device)
    # d(im0)
    res = d(im0)# res x y x y conf cls
    ldms = res[0]

    for ldm in ldms.tolist():
        w_c = int((ldm[2] - ldm[0]) / 4)
        h_c = int((ldm[3] - ldm[1]) / 4)

        n_x1 = max(int(ldm[0] - w_c), torch.tensor(0).to(device))
        n_y1 = max(int(ldm[1] - h_c), torch.tensor(0).to(device))
        n_x2 = min(int(ldm[2] + w_c), torch.tensor(im0.shape[1]).to(device))
        n_y2 = min(int(ldm[3] + h_c), torch.tensor(im0.shape[0]).to(device))
        crop = im0[n_y1:n_y2,n_x1:n_x2]
        # cv2.imshow('crop', crop)
        # cv2.waitKey(0)
        cls = c(crop)

        print(cls)
    print(res)