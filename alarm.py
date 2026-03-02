import pygame
import os

class alarm_det:
    def __init__(self, sound_file, volume=0.8):
        pygame.mixer.init()
        self.sound_file = sound_file
        self.volume = volume
        self.sound = None
        self.playing = False

        if os.path.exists(sound_file):
            self.sound = pygame.mixer.Sound(sound_file)
            self.sound.set_volume(volume)
        else:
            print("WARNING: Cannot Find alarm.wav")

    def start_alarm(self):
        if self.sound and not self.playing:
            self.sound.play(-1) # -1 = infinite loop
            self.playing = True

    def stop_alarm(self):
        if self.sound and self.playing:
            self.sound.stop()
            self.playing = False

    def is_active(self):
        return self.playing
