'''
@Project ：yolov8 
@File    ：exort.py
@Author  ：yuk
@Date    ：2024/3/19 11:02 
description：
'''
import onnxruntime
ort_session = onnxruntime.InferenceSession("/data0/zjt/yolov8/expo/best.onnx",
providers=['CUDAExecutionProvider'])
print(ort_session.get_providers())