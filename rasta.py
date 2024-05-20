import numpy as np
import scipy.signal
import librosa

class RASTA:
    def __init__(self, sample_rate, data):
        self.sample_rate = sample_rate        
        if data.ndim == 2:
            self.data = data.mean(axis=1).astype(float)
        else:
            self.data = data.astype(float)

    def filtering(self, filter_order=2, low_cut=0.1, high_cut=0.5, n_fft=128):
        # 音声信号の長さを確認し、短すぎる場合はゼロパディングを追加        
        if len(self.data) < n_fft:
            print(f"Signal length ({len(self.data)}) is shorter than n_fft ({n_fft}). Returning 0.")
            return np.zeros(self.data.shape)

        
        
        # STFTを計算
        stft = np.abs(librosa.stft(self.data, n_fft=n_fft))

        # 対数スペクトルを計算
        log_spectrum = np.log(stft + 1e-10)
        
        # バンドパスフィルタの設計
        b, a = scipy.signal.butter(filter_order, [low_cut, high_cut], btype='band')
        
        # フィルタリング前の信号長確認
        signal_length = log_spectrum.shape[1]
        padlen = 3 * max(len(b), len(a))  # scipy.signal.filtfiltのパディング長さ

        
        # フィルタの適用
        if signal_length > padlen:
            filtered_log_spectrum = scipy.signal.filtfilt(b, a, log_spectrum, axis=1)
        else:
            print(f"The length of the input vector x must be greater than padlen ({padlen}). Returning 0.")
            return np.zeros(self.data.shape)
        
        # 逆対数圧縮
        filtered_spectrum = np.exp(filtered_log_spectrum)
        
        # 逆STFTを計算
        filtered_signal = librosa.istft(filtered_spectrum)
        
        return filtered_signal
    

if __name__ == "__main__":
    import pathlib
    import scipy.io.wavfile as wav
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/swallowing/dataset/washino/swallowing/swallowing2.wav')
    sample_rate, data = wav.read(path)
    swallowing1 = RASTA(sample_rate, data)
    spectrogram = swallowing1.filtering()
    print(spectrogram.shape)
