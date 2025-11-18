import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

# ==============================
# 1. Настройки
# ==============================
SAMPLE_RATE = 16000 # Стандартная частота дискретизации
DURATION = 3 # Длительность аудио в секундах (обрезаем или дополняем до этого)
NUM_SAMPLES = SAMPLE_RATE * DURATION
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================
# 2. Обработка Аудио (Трансформация в спектрограмму)
# ==============================
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, target_sample_rate, num_samples, device):
        self.file_paths = file_paths
        self.labels = labels
        self.device = device
        
        # Трансформация: Аудио волна -> Мел-спектрограмма (картинка)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=64,  # Высота "картинки"
            n_fft=1024,
            hop_length=512
        ).to(device)
        
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.file_paths)

    def _cut_if_necessary(self, signal):
        # Если аудио длиннее, обрезаем
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        # Если аудио короче, дополняем тишиной (нулями)
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing = self.num_samples - length_signal
            last_dim_padding = (0, num_missing)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        # Приводим все аудио к одной частоте (16kHz)
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        # Если стерео -> делаем моно
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Загружаем
        signal, sr = torchaudio.load(path)
        signal = signal.to(self.device)
        
        # Препроцессинг
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        
        # Превращаем в спектрограмму
        spec = self.mel_spectrogram(signal)
        
        # Переводим в децибелы (логарифмирование) для лучшего обучения
        spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(spec)
        
        return spec, torch.tensor(label, dtype=torch.long)

# ==============================
# 3. Простая CNN модель для аудио
# ==============================
class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Вход: 1 канал (моно), выход: 16 каналов
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * (64//8 + 1) * (94//8 + 1), num_classes) 
        # Примечание: Размеры Linear слоя зависят от IMG_SIZE и параметров свертки.
        # Для продакшена лучше использовать AdaptiveAvgPool2d перед Flatten.
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5)) # Принудительно сжимает до 5x5
        self.final_linear = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        logits = self.final_linear(x)
        return logits

# Инициализация (как и в примере с фото)
# model = SimpleAudioCNN(num_classes=2).to(DEVICE)
# train_loader = DataLoader(...)
