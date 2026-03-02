import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class prog_ui:
    def __init__(self, title="Drowsiness Detector"):
        self.window = tk.Tk()
        self.window.title(title)

        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.running = False # this is to check if the detection block in main file is running?
        self.close_flag = False # this is to check if the user clicked close option?

        self.setup_widgets()

    def setup_widgets(self):
        main = ttk.Frame(self.window, padding="10")
        main.grid(row=0, column=0)

        self.vid_label = ttk.Label(main)
        self.vid_label.grid(row=0, column=0, columnspan=3, pady=10)

        status = ttk.LabelFrame(main, text="Status", padding="10")
        status.grid(row=1, column=0, columnspan=3, pady=5)

        self.ear_text = ttk.Label(status, text="EAR: --", font=("Arial", 12))
        self.ear_text.grid(row=0, column=0, padx=10)

        self.statusText = ttk.Label(status, text="Status: Ready", font=("Arial", 12, "bold"))
        self.statusText.grid(row=0, column=1, padx=10)

        self.alarm_text = ttk.Label(status, text="Alarm: OFF", foreground="green", font=("Arial", 12))
        self.alarm_text.grid(row=0, column=2, padx=10)

        buttons = ttk.Frame(main)
        buttons.grid(row=2, column=0, columnspan=3, pady=10)

        self.start_btn = ttk.Button(buttons, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.quit_btn = ttk.Button(buttons, text="Quit", command=self.close)
        self.quit_btn.grid(row=0, column=2, padx=5)

    def update_video(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # this blocl is to convert opencv image to PIL Image
        img = Image.fromarray(rgb)

        imgtk = ImageTk.PhotoImage(image=img)

        self.vid_label.imgtk = imgtk
        self.vid_label.configure(image=imgtk)

    def update_status(self, ear, drowsy, alarm):
        self.ear_text.config(text="EAR: " + str(round(ear, 3)))

        if drowsy:
            self.statusText.config(text="Status: DROWSY", foreground="red")
        else:
            self.statusText.config(text="Status: Alert", foreground="green")

        if alarm:
            self.alarm_text.config(text="Alarm: ON", foreground="red")
        else:
            self.alarm_text.config(text="Alarm: OFF", foreground="green")

    def start(self):
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def close(self):
        self.close_flag = True
        self.running = False
        self.window.quit()

    def update(self): #This block keeps the GUI responsive before this block the gui was getting freezed
        self.window.update()

    def is_closed(self):
        return self.close_flag
