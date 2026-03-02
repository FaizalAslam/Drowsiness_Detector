import logging
from datetime import datetime

class DataLogger:
    def __init__(self, log_file="log_details.txt"):
        self.log_file = log_file

        self.logger = logging.getLogger('Drowsiness-Detector')
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)
        separator = "=" * 80
        self.logger.info(separator)
        self.logger.info("Session Started")
        self.logger.info(separator)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def log_detection(self, ear, threshold, is_drowsy, frame_num):
        msg = f"DETECTION | Frame:{frame_num} | EAR:{round(ear, 4)} | Threshold:{round(threshold, 4)} | Drowsy:{is_drowsy}"
        self.logger.info(msg)

    def log_event(self, event_type, details):
        msg = "EVENT | Type:" + event_type + " | Details:" + details
        self.logger.info(msg)

    def log_stats(self, stats):
        separator = "=" * 80

        self.logger.info(separator)
        self.logger.info("SESSION DETAILS")

        for key in stats:
            value = stats[key]
            self.logger.info("  " + key + ": " + str(value))

        self.logger.info(separator)
