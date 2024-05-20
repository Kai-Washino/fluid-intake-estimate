import tensorflow as tf
from tensorflow.keras.layers import Masking, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .dence_net import DenceNet

class MLP(DenceNet): 
    def __init__(self, time_range = 30000):         
        self.model = Sequential([
            Masking(mask_value=0.0, input_shape=(time_range,)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='adam', 
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error'])



if __name__ == "__main__":
    from .variable_data_set import VariableDataSet
    import pathlib    

    data = VariableDataSet(130, scale=0, time_range=30000)
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/fluid_intake/dataset/ibuki')
    csv_path = path / 'ibuki.csv'    
    data.csv_to_dataset(path, csv_path, 0, signal_processing='rasta')
    print(data.X[0].max())
    model = MLP()
    model.training(data.X, data.y, 1, 16)