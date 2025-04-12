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