import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime

from adaptive_threshold import ThresholdAdapter
from logger import DataLogger

EAR_THRESHOLD = 0.21 #Found this value in the paper
DROWSY_FRAMES = 60
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

ALARM_FILE = "alarm.wav"
ALARM_VOLUME = 1.0 # max volume

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

WINDOW_TITLE = "Drowsiness Detection System"


def dist_bw(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def cal_ear(eye_pts):
    """
    Eye Aspect Ratio from Soukupova and Cech's formula:
    - 2 vert. dist
    - 1 hori. dist
    - EAR value = (vertical1 + vertical2) / (2 * horizontal)
    """
    v1 = dist_bw(eye_pts[1], eye_pts[5])
    v2 = dist_bw(eye_pts[2], eye_pts[4])
    h = dist_bw(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h)

def get_eye_coords(face_landmarks, idx_list):
    pts = []
    for i in idx_list:
        lm = face_landmarks.landmark[i]
        pts.append((lm.x, lm.y))
    return pts


class EyeDetector:
    #This are the landmarks taken from mediapipe documents, six points covering two corners, two points between these corners for both upper and lower eyelids
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        self.log = DataLogger()

        #Initialziing the facemesh from here, only the drive needs to be detected
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        self.adaptive = ThresholdAdapter(self.log)

        self.drowsy_count = 0
        self.frame_count = 0
        self.is_drowsy = False
        self.was_drowsy = False
        # Using deque with maxlen so I dont need to manually manage size
        self.ear_values = deque(maxlen=100)

        self.session_start = datetime.now()

        self.log.info("Detector initialized")

    def process_frame(self, frame):
        self.frame_count = self.frame_count + 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)
        ear = 0.0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            left_eye = get_eye_coords(face_landmarks, self.LEFT_EYE)
            right_eye = get_eye_coords(face_landmarks, self.RIGHT_EYE)

            left_ear = cal_ear(left_eye)
            right_ear = cal_ear(right_eye)

            ear = (left_ear + right_ear) / 2.0

            self.ear_values.append(ear)

            self.adaptive.update(ear, self.drowsy_count == 0)

            threshold = self.adaptive.get_threshold()

            if ear < threshold:
                self.drowsy_count += 1

                if self.drowsy_count == 10:
                    self.log.warning("Drowsy frames: " + str(self.drowsy_count))
            else:
                if self.drowsy_count > 0:
                    self.log.info("Eyes opened - counter reset")
                self.drowsy_count = 0
                self.is_drowsy = False

            if self.drowsy_count >= DROWSY_FRAMES: #This block is for the application to enable/disable the drowsiness and also check if its over the trheshold
                if not self.was_drowsy:
                    self.log.log_event("DROWSY_DETECTED",
                        "EAR=" + str(round(ear, 4)) + ", Frame=" + str(self.frame_count))
                self.is_drowsy = True
                self.was_drowsy = True
            else:
                if self.was_drowsy:
                    self.log.log_event("ALERT", "User became alert")
                self.was_drowsy = False

            if self.frame_count % 30 == 0: #Taking 30 frames a second because most camera supports 30fps
                self.log.log_detection(ear, threshold, self.is_drowsy, self.frame_count)

            self.eye_locator(frame, left_eye, right_eye)

            cv2.putText(frame, "EAR: " + str(round(ear, 3)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Threshold: " + str(round(threshold, 3)), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.is_drowsy:
                cv2.putText(frame, "DROWSY!", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return frame, ear, self.is_drowsy

    def eye_locator(self, frame, left_eye, right_eye):
        h, w = frame.shape[:2]

        for pt in left_eye:
            x = int(pt[0] * w)
            y = int(pt[1] * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        for pt in right_eye:
            x = int(pt[0] * w)
            y = int(pt[1] * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def save_data(self):
        duration = (datetime.now() - self.session_start).total_seconds()

        stats = {}
        stats['duration_seconds'] = round(duration, 2)
        stats['total_frames'] = self.frame_count

        if len(self.ear_values) > 0: #Transferring everyhting in numpy for future calculations and research
            ear_array = np.array(list(self.ear_values))
            stats['mean_ear'] = round(np.mean(ear_array), 4)
            stats['std_ear'] = round(np.std(ear_array), 4)
            stats['min_ear'] = round(np.min(ear_array), 4)
            stats['max_ear'] = round(np.max(ear_array), 4)

        self.log.log_stats(stats)
