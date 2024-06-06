import numpy as np
import pandas as pd
import pathlib
import sys
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from .data_set import DataSet
from .rasta import RASTA

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
        self.scale = scale
        
        if scale == 0:
            self.X = np.zeros((num_samples, self.time_range))        
        elif dimension is None:
            self.X = np.zeros((num_samples, self.time_range, self.scale))
        else:
            self.X = np.zeros((num_samples, self.dimension, scale))
        self.y = np.zeros(num_samples)     
        self.length = []

    def add_to_dataset(self, i, data, y):            
        if isinstance(data, tuple):
            spectrogram = np.abs(data)
        else:
            spectrogram = data         
        
        
        
        if len(spectrogram) == 0:
            print("Warning: No data available for signal processing.") 
            print(i)
            print(spectrogram)
            return 
                
        scaler_X = MinMaxScaler()        
        if spectrogram.ndim == 1:
            spectrogram = spectrogram.reshape(-1, 1)  # 1次元配列を2次元配列に変換               
        normalized_spectrogram = scaler_X.fit_transform(spectrogram)        
                              
        if self.dimension is None:           
            data = self.trim_or_pad(normalized_spectrogram)              
            data = data.reshape(self.time_range, self.scale)            
        else:
            data = self.pca(normalized_spectrogram)
            data = data.reshape(self.time_range, self.dimension)

            
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
            self.length.append(wav.length)            
            if signal_processing == 'wavelet':
                wavdata = Wavelet(wav.sample_rate, wav.trimmed_data, )
                coefficients, _ =  wavdata.generate_coefficients()                        
                self.add_to_dataset(start_num + index, coefficients, row['intake_volume'])
            elif signal_processing == 'fft':
                wavdata = FFT(wav.sample_rate, wav.trimmed_data, )
                spectrogram = wavdata.generate_spectrogram()                
                self.add_to_dataset(start_num + index, spectrogram,  row['intake_volume'])            
            elif signal_processing == 'rasta':
                wavdata = RASTA(wav.sample_rate, wav.trimmed_data, )
                filtered_signal = wavdata.filtering()
                self.add_to_dataset(start_num + index, filtered_signal,  row['intake_volume'])
            elif signal_processing == 'No':
                if len(wav.trimmed_data.shape) > 1:
                    wav.trimmed_data = wav.trimmed_data.mean(axis=1)                
                self.add_to_dataset(start_num + index, wav.trimmed_data,  row['intake_volume'])
                
            else:
                print("name is not define")


            
    def trim_or_pad(self, data):
        if len(data.shape) == 1:
            current_length = data.shape[0]
            if current_length > self.time_range:
                # トリミング            
                trimmed_data = data[:self.time_range]                
                return trimmed_data
            elif current_length < self.time_range:
                # パディング
                padding_length = self.time_range - current_length
                padded_data = np.pad(data, (0, padding_length), mode='constant', constant_values=0)                
                return padded_data
            else:
                # そのまま返す
                return data

        else:    
            current_length = data.shape[1]        
            if current_length > self.time_range:
                # トリミング            
                trimmed_data = data[:, :self.time_range]       
                return trimmed_data
            elif current_length < self.time_range:
                # パディング
                padding_length = self.time_range - current_length
                padded_data = np.pad(data, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
                return padded_data
            else:
                # そのまま返す
                return data  

if __name__ == "__main__":    
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/fluid_intake/dataset/ibuki')
    csv_path = path / 'ibuki.csv'
    # data = VariableDataSet(30, scale=222)
    # data.csv_to_dataset(path, csv_path, 0, signal_processing='fft')   
    data = VariableDataSet(30, scale=0)
    data.csv_to_dataset(path, csv_path, 0, signal_processing='rasta')    
    print(data.X.shape)
