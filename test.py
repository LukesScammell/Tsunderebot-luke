from TTS.api import TTS
import tempfile
import pygame
import os

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
    tts.tts_to_file(text="Hello world!", file_path=temp_wav.name)
    temp_path = temp_wav.name

pygame.mixer.init()
pygame.mixer.music.load(temp_path)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

os.remove(temp_path)
