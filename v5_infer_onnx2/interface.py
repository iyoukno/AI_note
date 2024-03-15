'''
@Project ：detect fire and smoke
@File    ：det_onnx.py
@Author  ：yuk
@Date    ：2024/3/14 10:35 
description：
'''

import sys
sys.path.append(r'/data0/zjt/v5_infer_onnx2/pac')


from pac.det.det_onnx import *
onnx_ph = r'/data0/zjt/v5_infer_onnx/best.onnx'
img_ph = r'/data0/zjt/v5_infer_onnx/test.jpg'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# im0 = cv2.imread(img_ph)
if __name__ == '__main__':
    d = Detect_by_onnx(onnx_path=onnx_ph, devicestr=device)
    # d(im0) 或 d(img_ph)
    res = d(img_ph)# res x y x y conf cls
    print(res)