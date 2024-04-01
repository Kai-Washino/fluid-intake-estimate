import numpy as np
import cv2  # OpenCVライブラリ
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import pathlib

import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
device_dir = os.path.dirname(parent_dir)
swallowing_recognize_dir = os.path.join(device_dir, 'swallowing')
print(swallowing_recognize_dir)
sys.path.append(swallowing_recognize_dir)
import swallowing_recognition
from swallowing_recognition.wavelet import Wavelet
from swallowing_recognition.audio import Audio

class DataSet:
    def __init__(self, num_samples, img_height, img_width, channels,):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.X = np.zeros((num_samples, img_height, img_width, channels))
        self.y = np.zeros(num_samples)

    def add_to_dataset(self, i, coefficients, y):
        spectrogram = np.abs(coefficients)
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val)
        resized_spectrogram = cv2.resize(normalized_spectrogram, (self.img_width, self.img_height))
        resized_spectrogram_uint8 = (resized_spectrogram * 255).astype(np.uint8)

        # グレースケール画像をRGBに変換
        resized_spectrogram_rgb = cv2.cvtColor(resized_spectrogram_uint8, cv2.COLOR_GRAY2RGB)
    
        # データセットに追加
        self.X[i] = resized_spectrogram_rgb
        self.y[i] = y
    
    def get_wav_files(self, directory):
        wav_files = []
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                wav_files.append(filename)
        return wav_files

    def folder_to_dataset(self, folder_name, y, start_num):        
        file_names = self.get_wav_files(folder_name)
        for i, file_name in enumerate(file_names):
            wav = Audio(folder_name / file_name)
            wavdata = Wavelet(wav.sample_rate, wav.trimmed_data, )
            coefficients, _ =  wavdata.generate_coefficients()
            self.add_to_dataset(start_num + i, coefficients, y)

    def print_label(self): 
        print(self.y)

    def print_data(self):
        print(self.data)

if __name__ == "__main__":
    print("hello")
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2023測定デバイス/swallowing/dataset/washino/voice/voice1.wav')
    wav1 = Audio(path)
    swallowing1 = Wavelet(wav1.sample_rate, wav1.trimmed_data, )
    coefficients, _ =  swallowing1.generate_coefficients()
    data = DataSet(1, 224, 224, 3)
    y = np.array(0.2)
    data.add_to_dataset(0, coefficients, y)
    print(data.X.shape)
    print(data.X[0][1][100][0])
    print(data.y)
    
