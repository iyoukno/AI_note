'''
@Project ：v5_infer_onnx 
@File    ：predict.py
@Author  ：yuk
@Date    ：2024/4/24 16:27 
description：
'''
import argparse
import os
import platform
import sys
from pathlib import Path
sys.path.append(r'D:\project\yolov5_infer\pac')
import torch
import torch.nn.functional as F
from ultralytics.utils.plotting import Annotator

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, print_args, strip_optimizer)
from utils.torch_utils import select_device, smart_inference_mode

class Classfier:
    def __init__(self, model_path, device, half=False, dnn=False, data=None, imgsz=(64,64)):
        self.half = half,  # use FP16 half-precision inference
        self.dnn = dnn,  # use OpenCV DNN for ONNX inference

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(model_path, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # self.__dict__.update(locals())
    def __call__(self, x):
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        # pre_handle img
        im = classify_transforms(self.imgsz[0])(x)
        with dt[0]:
            im = torch.Tensor(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            results = self.model(im)

        # Post-process
        with dt[2]:
            pred = F.softmax(results, dim=1)  # probabilities

        # print(pred.argmax().item())
        return pred.argmax().item()

if __name__ == '__main__':
    # sys.path.append(r'D:\project\yolov5_infer\pac')
    data1 = cv2.imread('data/0072.jpg')
    classifier = Classfier(r'D:\project\yolov5_infer\pac\classify\models\best.pt', device="cpu")
    classifier(data1)