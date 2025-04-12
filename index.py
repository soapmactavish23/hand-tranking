import cv2
import mediapipe as mp
import numpy as np
import time

confidence = float
webcam_image = np.ndarray
rbg_tuple = tuple[int, int, int]


class Detector:
    def __init__(self,
                 mode: bool = False,
                 number_hands: int = 2,
                 model_complexity: int = 1,
                 min_detec_confidence: confidence = 0.5,
                 min_tracking_confidence: confidence = 0.5):
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence

        #Inicializar o Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_num_hands,
            self.complexity,
            self.detection_con,
            self.tracking_con)
        self.tip_ids = [4, 8, 12, 16, 20]

if __name__ == '__main__':
    Detec = Detector()

    capture = cv2.VideoCapture(0)

    while True:
        _, img = capture.read()

        cv2.imshow('Camera da Azimov', img)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break