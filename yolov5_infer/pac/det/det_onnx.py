'''
@Project ：detect fire and smoke
@File    ：det_onnx.py
@Author  ：yuk
@Date    ：2024/3/14 10:35 
description：
'''
import torch
import numpy as np
from pathlib import Path
from det.pack import common
from ultralytics.utils.plotting import Annotator, colors
import platform
from utils.general import (
    Profile,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,
)
from utils.augmentations import (
    letterbox,
)

class Detect_by_onnx():
    def __init__(self, onnx_path, path=None, devicestr='cpu',):
        device = torch.device(devicestr)
        imgsz = (640, 640)
        stride = 32
        bs = 1
        conf_thres = 0.35,  # confidence threshold
        iou_thres = 0.45,  # NMS IOU threshold
        max_det = 1000,
        save_dir = None
        augment = False
        classes = None
        agnostic_nms = False
        visualize = False
        hide_conf = False
        view_img = True
        ox_model = common.Load_onnx(onnx_path, device=device)
        stride, names, pt = ox_model.stride, ox_model.names, ox_model.pt
        vid_path, vid_writer = [None] * bs, [None] * bs
        # dataset = common.LoadImages(path, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        self.__dict__.update(locals())

    def run(self, *args, **kwargs):
        '''
        可以读取一个文件夹的所有图片和视频，使用这个方法时将common.LoadImages放开（测试时使用，不用于接口）
        Args:
            *args:
            **kwargs:

        Returns:

        '''
        self.ox_model.warmup(imgsz=(1 if self.pt or self.ox_model.triton else self.bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))
        for path, im, im0s, vid_cap, s in self.dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.ox_model.device)
                im = im.half() if self.ox_model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if self.ox_model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)
            # Inference
            with dt[1]:
                visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                if self.ox_model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = self.ox_model.proces(image).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, self.ox_model.proces(image).unsqueeze(0)), dim=0)
                    pred = [pred, None]
                else:
                    pred = self.ox_model.proces(im)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres[0], self.iou_thres[0], self.classes, self.agnostic_nms, max_det=self.max_det[0])

            # show
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(self.dataset, "frame", 0)

                p = Path(p)  # to Path

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=1, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = self.names[c] if self.hide_conf else f"{self.names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = (self.names[c] if self.hide_conf else f"{self.names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))


                # Stream results
                im0 = annotator.result()
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)

        print(pred)# xyxy, conf, cls

    def __call__(self, source):
        '''
        one imge by path or ndarray（interface）
        Args:
            source: path or ndarray

        Returns:

        '''
        if isinstance(source, str) and Path(source).suffix in (".jpg", "png"):
            p = source
            im0 = cv2.imread(source)  # BGR

        else:
            im0 = source
        im = letterbox(im0, self.imgsz, stride=self.stride, auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        self.ox_model.warmup(imgsz=(1 if self.pt or self.ox_model.triton else self.bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (
        Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))

        with dt[0]:
            im = torch.from_numpy(im).to(self.ox_model.device)
            im = im.half() if self.ox_model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if self.ox_model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
        # Inference
        with dt[1]:
            if self.ox_model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = self.ox_model.proces(image).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, self.ox_model.proces(image).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = self.ox_model.proces(im)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres[0], self.iou_thres[0], self.classes, self.agnostic_nms,
                                       max_det=self.max_det[0])

        # show
        for i, det in enumerate(pred):  # per image
            # seen += 1
            # im0 = im0.copy()
            # annotator = Annotator(im0, line_width=1, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                # for *xyxy, conf, cls in reversed(det):
                #     c = int(cls)  # integer class
                #     label = self.names[c] if self.hide_conf else f"{self.names[c]}"
                #     confidence = float(conf)
                #     confidence_str = f"{confidence:.2f}"

                    # if self.view_img:  # Add bbox to image
                    #     c = int(cls)  # integer class
                    #     label = (self.names[c] if self.hide_conf else f"{self.names[c]} {conf:.2f}")
                    #     annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            # im0 = annotator.result()
            # cv2.imshow('img', im0)
            # cv2.waitKey(0)

        # print(pred)  # xyxy, conf, cls
        return pred

# data = torch.randn(1,3,640,640).numpy()
# pred = ox_model.proces(data)
# print(pred)
# onnx_ph = r'Z:\zjt\yolov5-master\runs\train\exp5\weights\best.onnx'
# img_ph = r'Z:\zjt\yolov5-master\expo\test.jpg'
# im0 = cv2.imread(img_ph)
# if __name__ == '__main__':
#     d = Detect_by_onnx(onnx_path=onnx_ph,path=img_ph)
#     d.run()
#     # d(im0)