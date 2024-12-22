import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import pickle
import time
import numpy as np
import keyboard
from collections import deque
import sys
import pygame

pygame.init()

MAX_MFCC_LENGTH=100
MFCC_DIM=40
freq = 16000
duration = 1
channels = 1
threshold=0.7
audio_buffer =  deque(maxlen=freq * duration)
chunk_size=int(freq/4)
word_detected=False
text_color = (0,0,0)
background_color = (255, 255, 255) 
running=True
width, height = 1500, 700
larger_font = pygame.font.Font("helvetica.ttf", 180)
smaller_font = pygame.font.Font("helvetica.ttf", 74)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Speech recognition")

def convert_to_mfcc(recording):
    recording = np.squeeze(recording)
    mel_spectrogram = librosa.feature.melspectrogram(y=recording, sr=freq, n_fft=512, hop_length=160, n_mels=MFCC_DIM)
    mel_spectrogram += 1e-6 
    log_mel_spectrogram = np.log(mel_spectrogram)
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=MFCC_DIM)

    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, MAX_MFCC_LENGTH - mfcc.shape[1]))), mode='constant')[:, :MAX_MFCC_LENGTH]
    mfcc = mfcc.astype(np.float16)  

    mfcc = mfcc[1:]
    mfcc = np.expand_dims(mfcc, axis=0)  
    return mfcc

def audio_callback(indata, frames, time, status):
    global audio_buffer
    global word_detected

    if status:
        screen.fill(background_color)

    
    audio_buffer.extend(indata.flatten())

    if len(audio_buffer) == freq * duration:
        mfcc = convert_to_mfcc(np.array(audio_buffer))
        predictions = model.predict(mfcc, verbose=0)

        probabilities = predictions[0]
        class_labels = label_encoder.inverse_transform(range(len(probabilities)))

        most_likely_index = np.argmax(probabilities)
        most_likely_probability = probabilities[most_likely_index]
        most_likely_label = class_labels[most_likely_index]

        if most_likely_probability > threshold and (word_detected):
            audio_buffer.clear() 
            text = f"{most_likely_label} {100*most_likely_probability:.1f}%"
            text_surface = larger_font.render(text, True, text_color)
            text_rect = text_surface.get_rect(center=(width // 2, height // 2)) 
            screen.fill(background_color)
            screen.blit(text_surface, text_rect)


        elif most_likely_probability > threshold:
            word_detected =True
        else:
            word_detected = False
    pygame.display.flip()

try:
    model = tf.keras.models.load_model("nfft512_BNall_2x32-3x3_max2_2x64-3x3_max2_2x128-3x3_max2_reshape-13x640_biLSTM128_256_0.5drop_6ep.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("Label encoder loaded.")

try:
    with sd.InputStream(
        callback=audio_callback, channels=1, samplerate=freq, blocksize=chunk_size
    ):
        while not keyboard.is_pressed("esc") and running:
            sd.sleep(250) 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        print("Recording stopped.")
except KeyboardInterrupt:
    print("Recording stopped.")
pygame.quit()
sys.exit()

