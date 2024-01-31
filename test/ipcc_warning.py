'''
@Project ：test 
@File    ：ipcc_warning.py
@Author  ：yuk
@Date    ：2024/1/31 11:44 
description：handle warning: iCCP: known incorrect sRGB profile
'''
import os
from tqdm import tqdm
import cv2
from skimage import io
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
path = r"D:\datasets\FallDetection\images\\"
fileList = os.listdir(path)
for i in tqdm(fileList[9438:]):
    # print(i)
    try:
        image = io.imread(path+i)  # image = io.imread(os.path.join(path, i))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        cv2.imencode('.jpg',image)[1].tofile(path+i)
    except:
        print(i)


# image = io.imread(path)  # image = io.imread(os.path.join(path, i))
# image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
# cv2.imencode('.jpg',image)[1].tofile(path)