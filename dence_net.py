from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

class DenceNet:
    def __init__(self):
        self.base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3))
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1, activation=None)(x)  # 回帰
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        # モデルのコンパイル
        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error'])
        
    def training(self, X_train, y_train, epochs, batch_size, early_stopping = None, model_checkpoint = None):
        if early_stopping == None and model_checkpoint == None:
            self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, batch_size= batch_size)
        elif early_stopping == None:
            self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, batch_size= batch_size, callbacks=[model_checkpoint])
        elif model_checkpoint == None:
            self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, batch_size= batch_size, callbacks=[early_stopping])
        else:
            self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, batch_size= batch_size, callbacks=[early_stopping, model_checkpoint])

    def evaluate(self, X_test, y_test, print_value = False):
        # モデルの評価を実行し、損失を取得（評価指標を複数指定していればそれらも取得される）
        test_loss, *test_metrics = self.model.evaluate(X_test, y_test)
        print("Test loss: ", test_loss)
        for i, metric in enumerate(test_metrics):
            print(f"Test metric {i}: ", metric)
        
        # モデルによる予測
        self.predictions = self.model.predict(X_test)
        
        r2 = r2_score(self.predictions, y_test)
        print("r2: ", r2)
        
        mae = mean_absolute_error(self.predictions, y_test)
        print("MAE: ", mae)
        
        if print_value:
        # 予測値と実際の値を表示（オプション）
            for i in range(len(X_test)):
                print(f"サンプル {i}: 正解 = {y_test[i]}, 予測 = {self.predictions[i]}")
    
    def save(self, file_name):
        self.model.save(file_name)

if __name__ == "__main__":
    from .data_set import DataSet
    import pathlib
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2023測定デバイス/flueid_intake/dataset/fake_data')
    csv_path = path / 'fake_data.csv'
    data = DataSet(100, 224, 224, 3)
    data.csv_to_dataset(path, csv_path, 0)
    model = DenceNet()
    model.training(data.X, data.y, 1, 32)