'''
@Project ：test 
@File    ：save_video.py
@Author  ：yuk
@Date    ：2024/1/19 9:10 
description：读取视频保存为图片或视频
'''
import cv2


def save_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    totalFrames = int(cap.get(7))
    vid_writer = cv2.VideoWriter('./new2.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
            vid_writer.write(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

path = r'C:\Users\liyz\Desktop\2.mp4'

save_video(path)