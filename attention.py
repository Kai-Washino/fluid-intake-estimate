import tensorflow as tf
from tensorflow.keras.layers import Masking, Dense, Input, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import numpy as np

from .dence_net import DenceNet

class SelfAttentionModel(DenceNet):
    def __init__(self, scale=1, time_range=5000, d_model=16, num_heads=2, num_layers=2, dff=128, dropout_rate=0.1):
        self.scale = scale
        self.time_range = time_range

       # 入力テンソルの定義
        inputs = Input(shape=(time_range, scale))  # (70000, 1) の形状の入力

        # エンベディング
        x = Dense(d_model)(inputs)  # Denseレイヤーを適用し、(70000, 16) の形状に変換
        x *= tf.math.sqrt(tf.cast(d_model, tf.float32))  # スケーリング

        # 位置エンコーディングの追加（省略可、必要に応じて追加）
        x += self.positional_encoding(time_range, d_model)

        for _ in range(num_layers):
            x = self.encoder_layer(x, d_model, num_heads, dff, dropout_rate)

        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=outputs)

        self.model.compile(optimizer='adam', 
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error'])

    def encoder_layer(self, x, d_model, num_heads, dff, dropout_rate):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn_output = Dense(dff, activation='relu')(out1)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        return out2

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

if __name__ == "__main__":
    from .variable_data_set import VariableDataSet
    import pathlib    

    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/fluid_intake/dataset/hikaru')
    csv_path = path / 'hikaru.csv'
    data = VariableDataSet(35, scale=1, time_range=5000)
    data.csv_to_dataset(path, csv_path, 0, signal_processing='No')
    print(data.X.shape)
    model = SelfAttentionModel(scale=1, time_range=5000)
    model.training(data.X, data.y, 1, 32)
