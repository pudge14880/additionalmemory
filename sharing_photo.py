import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os

# ==============================
# 1. Настройки (Config)
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LR = 1e-3
NUM_EPOCHS = 5
NUM_CLASSES = 2  # Например, кошки vs собаки
IMG_SIZE = 224

# ==============================
# 2. Датасет (Легко изменить источник данных)
# ==============================
class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        file_paths: список путей к картинкам
        labels: список меток (int)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Загрузка изображения
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# ==============================
# 3. Аугментации и Загрузчики
# ==============================
# Базовые трансформации для ResNet
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(), # Аугментация
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ПРИМЕР ЗАГЛУШКИ ДАННЫХ (Замените на реальные списки)
train_files = ["img1.jpg", "img2.jpg"] * 50 # Пути
train_labels = [0, 1] * 50                  # Метки
# Создаем датасет
train_dataset = CustomImageDataset(train_files, train_labels, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==============================
# 4. Модель (Легко заменяемая)
# ==============================
def get_model(num_classes):
    # Загружаем ResNet18 с весами ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Если нужно заморозить веса (Feature Extraction):
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Заменяем последний слой (голову) под наши классы
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

model = get_model(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# 5. Цикл обучения (Training Loop)
# ==============================
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Важно: здесь нужен try-except, если файлов "заглушек" нет на диске
    try:
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")
    except FileNotFoundError:
        print("Ошибка: Файлы изображений не найдены. Укажите реальные пути в train_files.")
        break
