import numpy as np
from collections import deque

class ThresholdAdapter:
    def __init__(self, logger=None, window_size=100, initial_threshold=0.21):
        self.logger = logger
        self.window_size = window_size
        self.threshold = initial_threshold # starting with standard value

        self.ear_history = deque(maxlen=window_size)
        self.open_samples = deque(maxlen=50)
        self.closed_samples = deque(maxlen=30)

        self.calibrated = False
        self.sample_count = 0

    def update(self, ear_val, is_open=True):
        self.ear_history.append(ear_val)
        self.sample_count = self.sample_count + 1

        if is_open and ear_val > 0.25:
            self.open_samples.append(ear_val)
        elif not is_open and ear_val < 0.20:
            self.closed_samples.append(ear_val)

        if self.sample_count == 50:
            if not self.calibrated:
                self.calibrate()

        if self.sample_count % 100 == 0:
            if self.calibrated:
                self.recalibrate()

    def calibrate(self): #this is to determine average EAR for differnet facial structure to get better results
        if len(self.open_samples) > 10 and len(self.closed_samples) > 5:
            """if len(self.open_samples) > 30 and len(self.closed_samples) > 10:
            #Tried to use this so that the system gets more data samples but the system went inconsistent,
            so tried different number,
            it seems like 10 and 5 are optimal giving 0.33 seconds for the system to adapt with open eyes
            and 0.167 seconds for the system to adapt with close eyes.

            Not really sure which number is the optimal number, currently using this because this allowed the system to work without any issues.
            """
            avg_open = sum(self.open_samples) / len(self.open_samples)
            avg_closed = sum(self.closed_samples) / len(self.closed_samples)

            new_threshold = (avg_open + avg_closed) / 2.0
            new_threshold -= 0.02

            if new_threshold < 0.15:
                new_threshold = 0.15
            if new_threshold > 0.30:
                new_threshold = 0.30

            self.threshold = new_threshold
            self.calibrated = True

            if self.logger:
                details = "Threshold="
                details = details + str(round(self.threshold, 4))
                details = details + ", Open="
                details = details + str(round(avg_open, 4))
                details = details + ", Closed="
                details = details + str(round(avg_closed, 4))
                self.logger.log_event("CALIBRATION", details)

    def recalibrate(self): #checked online because percentile should work better than mean in this context
        if len(self.ear_history) >= self.window_size:
            ear_list = []
            for val in self.ear_history:
                ear_list.append(val)

            sorted_ears = []
            for val in ear_list:
                sorted_ears.append(val)
            sorted_ears.sort()
            index_90 = int(90 * len(sorted_ears) / 100)
            index_10 = int(10 * len(sorted_ears) / 100)

            p90 = sorted_ears[index_90]
            p10 = sorted_ears[index_10]

            new_thresh = (p90 + p10) / 2.0
            alpha = 0.1 #This is to prevent sudden increase in threshold, so it calculates using 90% of the old value and then the new value only contibutes 10% for less false values
            old_threshold = self.threshold

            self.threshold = (alpha * new_thresh) + ((1 - alpha) * self.threshold)

            if self.logger:
                msg = "Threshold: "
                msg = msg + str(round(old_threshold, 4))
                msg = msg + " -> "
                msg = msg + str(round(self.threshold, 4))
                self.logger.log_event("RECALIBRATION", msg)

    def get_threshold(self):
        return self.threshold

    def get_confidence(self):
        conf = self.sample_count / 200.0
        if conf > 1.0:
            conf = 1.0
        return conf