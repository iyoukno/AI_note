'''
@Project ：yolov8 
@File    ：det_hand2.py
@Author  ：yuk
@Date    ：2024/3/18 10:33 
description：
'''
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# custom style
landmark_annotations = mp_drawing_styles.get_default_hand_landmarks_style()
connection_annotations = mp_drawing_styles.get_default_hand_connections_style()
landmark_annotations = mp_drawing.DrawingSpec(
    color=mp_drawing.RED_COLOR, thickness=-1, circle_radius=2)
connection_annotations = mp_drawing.DrawingSpec(
    color=(0,200,0), thickness=2)
# file = r"Z:\zjt\yolov8\data\test3.jpg"
# data = cv2.imread(file)

# For static images:
def hand_detect(array):
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:

        image = cv2.flip(array, 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        # print('Handedness:', results.multi_handedness)
        # if not results.multi_hand_landmarks:
        #     continue
        if results.multi_hand_landmarks:
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                # print('hand_landmarks:', hand_landmarks)
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                # )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_annotations,
                    connection_annotations)
            # cv2.imshow(
            #     'img', cv2.flip(annotated_image, 1))
            # cv2.waitKey(0)
            return cv2.flip(annotated_image, 1), results.multi_hand_landmarks
        else:
            return array, []

# hand_detect(data)
