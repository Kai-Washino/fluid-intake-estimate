import tensorflow as tf
from tensorflow.keras.layers import Masking, LSTM, Dense

from .dence_net import DenceNet

class LSTMmodel(DenceNet):
    def __init__(self, scale=1, time_range=70000, num_class=2, lstm_units=50):
        self.num_class = num_class
        self.scale = scale
        self.time_range = time_range
        self.model = tf.keras.models.Sequential([
            Masking(mask_value=0.0, input_shape=(time_range, scale)),            
            LSTM(lstm_units, return_sequences=True),  # 第1 LSTM層
            LSTM(lstm_units, return_sequences=True),  # 第2 LSTM層
            LSTM(lstm_units),  # 第3 LSTM層
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')  # 出力層
        ])

        self.model.compile(optimizer='adam', 
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error'])

if __name__ == "__main__":
    from .variable_data_set import VariableDataSet
    import pathlib    

    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/fluid_intake/dataset/hikaru')
    csv_path = path / 'hikaru.csv'
    data = VariableDataSet(35, scale=1)
    data.csv_to_dataset(path, csv_path, 0, signal_processing='No')
    print(data.X.shape)
    model = LSTMmodel()
    model.training(data.X, data.y, 1, 32)
