import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Masking, Flatten, Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .dence_net import DenceNet

class CNN(DenceNet): 
    def __init__(self, scale = 127, time_range = 70000, num_class = 2, start_filter = 8):
        self.num_class = num_class
        self.scale = scale
        self.time_range = time_range                
        self.model = tf.keras.models.Sequential([
            Masking(mask_value=0.0, input_shape=(time_range, scale)),            
            Conv1D(start_filter, 3, activation='relu'),  # 第1畳み込み層
            MaxPooling1D(2),  # 第1プーリング層
            Conv1D(start_filter * 2, 3, activation='relu'),  # 第2畳み込み層
            MaxPooling1D(2),  # 第2プーリング層
            Conv1D(start_filter * 2 * 2, 3, activation='relu'),  # 第3畳み込み層
            MaxPooling1D(3),  # 第3プーリング層
            Conv1D(start_filter * 2 * 2 * 2, 3, activation='relu'),  # 第4畳み込み層
            MaxPooling1D(1),  # 第4プーリング層
            Flatten(),  # データのフラット化
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])

        self.model.compile(optimizer='adam', 
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error'])


if __name__ == "__main__":
    from .variable_data_set import VariableDataSet
    import pathlib    

    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/fluid_intake/dataset/fake_data')
    csv_path = path / 'fake_data.csv'
    data = VariableDataSet(100)
    data.csv_to_dataset(path, csv_path, 0)
    print(data.X.shape)
    model = CNN()
    model.training(data.X, data.y, 1, 32)