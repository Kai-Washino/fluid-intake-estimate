import numpy as np
import pandas as pd
import pathlib
import sys
import os

from sklearn.decomposition import PCA

from .data_set import DataSet

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
device_dir = os.path.dirname(parent_dir)
swallowing_recognize_dir = os.path.join(device_dir, 'swallowing')
print(swallowing_recognize_dir)
sys.path.append(swallowing_recognize_dir)
import swallowing_recognition
from swallowing_recognition.wavelet import Wavelet
from swallowing_recognition.audio import Audio
from swallowing_recognition.fft import FFT

class VariableDataSet(DataSet):
    def __init__(self, num_samples, scale = 127, time_range = 70000, dimension = None):
        self.time_range = time_range
        self.dimension = dimension
        
        if dimension is None:
            self.X = np.zeros((num_samples, scale, self.time_range))
        else:
            self.X = np.zeros((num_samples, scale, self.dimension))
        self.y = np.zeros(num_samples)        

    def add_to_dataset(self, i, data, y):        
        if type(data) == tuple:
            spectrogram = np.abs(data)        
        else:
            spectrogram = data         
            
        if len(spectrogram) == 0:
            print("Warning: No data available for FFT.") 
            print(i)
            print(spectrogram)
            return 
        
        spectrogram = np.abs(data)        
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val)        
        
        if self.dimension is None:
            data = self.trim_or_pad(normalized_spectrogram)        
        else:
            data = self.pca(normalized_spectrogram)
            
        self.X[i] = data
        self.y[i] = y
   
    def pca(self, data):
        if self.dimension is None:
            return data
        else:
            pca = PCA(n_components= self.dimension)  # 100次元に削減
            transformed_data = pca.fit_transform(data)
            return transformed_data
    
    def csv_to_dataset(self, path, csv_path, start_num, signal_processing = "wavelet"):
        df = pd.read_csv(csv_path)        
        for index, row in df.iterrows():            
            wav = Audio(path / row['wav_file_name'])
            if signal_processing == 'wavelet':
                wavdata = Wavelet(wav.sample_rate, wav.trimmed_data, )
                coefficients, _ =  wavdata.generate_coefficients()                        
                self.add_to_dataset(start_num + index, coefficients, row['intake_volume'])
            elif signal_processing == 'fft':
                wavdata = FFT(wav.sample_rate, wav.trimmed_data, )
                spectrogram = wavdata.generate_spectrogram()
                self.add_to_dataset(start_num + index, spectrogram,  row['intake_volume'])


            
    def trim_or_pad(self, data):
        current_length = data.shape[1]        
        if current_length > self.time_range:
            # 70000以上の場合はトリミング            
            trimmed_data = data[:, :self.time_range]       
            return trimmed_data
        elif current_length < self.time_range:
            # 70000未満の場合はパディング
            padding_length = self.time_range - current_length
            padded_data = np.pad(data, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
            return padded_data
        else:
            # すでに70000の場合はそのまま返す
            return data  

if __name__ == "__main__":    
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/fluid_intake/dataset/ibuki')
    csv_path = path / 'ibuki.csv'
    data = VariableDataSet(30, scale=222)
    data.csv_to_dataset(path, csv_path, 0, signal_processing='fft')    
    print(data.X.shape)
