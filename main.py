import cv2

from detector import EyeDetector, CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, ALARM_FILE, ALARM_VOLUME, WINDOW_TITLE
from gui import prog_ui
from alarm import alarm_det

def main():
    print("Drowsiness Detection System")

    detector = EyeDetector()
    gui = prog_ui(WINDOW_TITLE)
    alarm = alarm_det(ALARM_FILE, ALARM_VOLUME)

    camera = cv2.VideoCapture(CAMERA_ID)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not camera.isOpened():
        print("ERROR: Camera is not initialized")
        return

    try:
        while not gui.is_closed():
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame")
                break

            # This Process will only work when detection is running
            if gui.running:
                frame, ear, drowsy = detector.process_frame(frame)

                if drowsy:
                    alarm.start_alarm()
                else:
                    alarm.stop_alarm()

                gui.update_video(frame)
                gui.update_status(ear, drowsy, alarm.is_active())
            else:
                gui.update_video(frame)
                gui.update_status(0.0, False, False)

            gui.update()

    except Exception as e:
        print("Error:", str(e))

    finally: #This block needs to work no matter how the application ended that's wht we are using finally block
        alarm.stop_alarm()
        detector.save_data()
        camera.release()
        print("Session Ended.")

if __name__ == "__main__":
    main()
