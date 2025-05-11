import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
import cv2
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


# 1. Кастомный датасет
class CenterDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Получаем список файлов (имена должны совпадать)
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

        # Проверка соответствия
        assert len(self.image_files) == len(self.mask_files), "Количество изображений и масок не совпадает"
        for img, msk in zip(self.image_files, self.mask_files):
            assert img == msk.replace('mask_', ''), f"Несоответствие имен: {img} vs {msk}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Загрузка изображения и маски
        img_path = self.images_dir / self.image_files[idx]
        mask_path = self.masks_dir / self.mask_files[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Находим центр объекта из маски
        y, x = np.where(mask > 0)
        if len(x) > 0 and len(y) > 0:
            center_x = x.mean() / mask.shape[1]  # Нормализуем к [0, 1]
            center_y = y.mean() / mask.shape[0]
        else:
            center_x, center_y = 0.5, 0.5  # Если объект не найден

        # Преобразования
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([center_x, center_y], dtype=torch.float32)


# 2. Модель для регрессии центра
class CenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.fc = nn.Linear(512, 2)  # 2 выхода: x и y координаты

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))  # Sigmoid для нормализации выхода [0, 1]


# 3. Функция обучения
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, centers in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            centers = centers.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, centers)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Валидация
        val_loss = evaluate_model(model, val_loader, criterion)

        # Сохраняем историю
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train'].append(epoch_loss)
        history['val'].append(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Сохраняем лучшую модель
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return history


def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, centers in loader:
            images = images.to(device)
            centers = centers.to(device)

            outputs = model(images)
            loss = criterion(outputs, centers)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


# 4. Основной код
if __name__ == "__main__":
    # Параметры
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LR = 0.001

    # Пути к данным
    images_dir = "training_dataset/cros/generate/output_dxf"
    masks_dir = "masks/cross"

    # Преобразования
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Создаем датасеты и загрузчики
    full_dataset = CenterDataset(images_dir, masks_dir, transform=transform)

    # Разделение на train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterNet().to(device)

    # Критерий и оптимизатор
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Обучение
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)

    # Визуализация обучения
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()


# 5. Скрипт для предсказания центра
def predict_center(model, image_path, transform):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if transform:
        image = transform(image)

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        center = model(image)
        center = center.squeeze().cpu().numpy()

    return center  # Возвращает нормализованные координаты [x, y]

