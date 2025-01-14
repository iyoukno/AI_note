'''
@Project ：test 
@File    ：check_res.py
@Author  ：yuk
@Date    ：2024/5/16 10:14 
description：
'''
import os

import cv2
import numpy as np
from tqdm import tqdm


def iou(boxes0: np.ndarray, boxes1: np.ndarray):
    """ 计算多个边界框和多个边界框的交并比

    Parameters
    ----------
    boxes0: `~np.ndarray` of shape `(A, 4)`
        边界框

    boxes1: `~np.ndarray` of shape `(B, 4)`
        边界框

    Returns
    -------
    iou: `~np.ndarray` of shape `(A, B)`
        交并比
    """
    A = boxes0.shape[0]
    B = boxes1.shape[0]

    xy_max = np.minimum(boxes0[:, np.newaxis, 2:].repeat(B, axis=1),
                        np.broadcast_to(boxes1[:, 2:], (A, B, 2)))
    xy_min = np.maximum(boxes0[:, np.newaxis, :2].repeat(B, axis=1),
                        np.broadcast_to(boxes1[:, :2], (A, B, 2)))

    # 计算交集面积
    inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, :, 0]*inter[:, :, 1]

    # 计算每个矩阵的面积
    area_0 = ((boxes0[:, 2]-boxes0[:, 0])*(
        boxes0[:, 3] - boxes0[:, 1]))[:, np.newaxis].repeat(B, axis=1)
    area_1 = ((boxes1[:, 2] - boxes1[:, 0])*(
        boxes1[:, 3] - boxes1[:, 1]))[np.newaxis, :].repeat(A, axis=0)

    return inter/(area_0+area_1-inter)


res_pth = r'X:\zjt\yolov8\res\det5\labels'
label_pth = r'X:\zjt\datasets\safety-helmet\VOC2028\labels'
# out = open(r'D:\project\selfUse\test\resTxt/pp_fall_t1.txt', mode='w')

img_path = r'X:\zjt\datasets\safety-helmet\VOC2028\images'
# 想要统计的类型
sta_cls = 1

sta_cls_pred_TP = 0
sta_cls_pred_FN = 0
pred_FP = 0

# 指定类型的个数
sta_cls_label = 0
# 指定类型以外的个数
sta_other_cls_label = 0


lll = os.listdir(res_pth)
# idx = lll.index('fyb_025.txt')
for i in tqdm(lll):
    # real_cls0 = 0
    # real_cls1 = 0
    # pred_cls0 = 0
    # pred_cls1 = 0
    # print(f'{i} {lll.index(i)}')
    img_name = i[:-4]+'.jpg'
    img = cv2.imread(os.path.join(img_path,img_name))
    img_w = img.shape[1]
    img_h = img.shape[0]
    label_f = open(os.path.join(label_pth, i), 'r')
    f = open(os.path.join(res_pth, i), 'r')
    label_lines = label_f.readlines()
    for line in label_lines:
        if int(line[0]) == sta_cls:
            sta_cls_label += 1
        if int(line[0]) != sta_cls:
            sta_other_cls_label += 1
        # print(line)
    lines = f.readlines()
    label_data = np.array([[np.array(float(i)) for i in x.strip().split(' ')] for x in label_lines])
    res_data = np.array([[np.array(float(i)) for i in x.strip().split(' ')] for x in lines])

    label_box = np.zeros(label_data.shape)
    res_box = np.zeros((res_data.shape[0],5))

    label_box[:, 1] = (label_data[:, 1] * img_w) - (label_data[:, 3] * img_w / 2)
    label_box[:, 2] = (label_data[:, 2] * img_h) - (label_data[:, 4] * img_h /2)
    label_box[:, 3] = (label_data[:, 1] * img_w) + (label_data[:, 3] * img_w /2)
    label_box[:, 4] = (label_data[:, 2] * img_h) + (label_data[:, 4] * img_h /2)

    res_box[:, 1] = (res_data[:, 1] * img_w) - (res_data[:, 3] * img_w / 2)
    res_box[:, 2] = (res_data[:, 2] * img_h) - (res_data[:, 4] * img_h / 2)
    res_box[:, 3] = (res_data[:, 1] * img_w) + (res_data[:, 3] * img_w / 2)
    res_box[:, 4] = (res_data[:, 2] * img_h) + (res_data[:, 4] * img_h / 2)

    label_f.close()
    f.close()
    # 算iou
    res = iou(label_box[:,1:], res_box[:,1:])

    len(res[res>0.35])
    iou_matched = res_data[np.where(res > 0.35)[1]]
    label_iou_matched = label_data[np.where(res > 0.35)[0]]
    for idx, res in enumerate(iou_matched):
        # 实际：正，预测：正
        if label_iou_matched[idx][0] == sta_cls and int(res[0] == sta_cls):
            sta_cls_pred_TP += 1
        # 实际：负，预测：正
        if label_iou_matched[idx][0] != sta_cls and int(res[0] == sta_cls):
            pred_FP += 1


    # str_info = f'{i[:-4]}.jpg\t{real_cls0}\t{real_cls1}\t{pred_cls0}\t{pred_cls1}\n'
    # out.write(str_info)
# out.close()

precision = sta_cls_pred_TP / (sta_cls_pred_TP + pred_FP)
recall = sta_cls_pred_TP / sta_cls_label

print('precision：',precision)
print('recall：',recall)

# box0 = np.array([
#     [5,5,10,10],
#     [15,15,20,20]
# ])
#
# box1 = np.array([
#     [6,6,12,12],
#     [21,21,27,27],
#     [17,17,25,25],
#     [5,5,12,12],
# ])
#
# res = iou(box0,box1)
# print(res)
#
# b = box1[np.nonzero(res)[1]][:,0] == box0[np.nonzero(res)[0]][:,0]
