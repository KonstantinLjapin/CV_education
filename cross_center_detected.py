import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms, models
import time
from datetime import datetime


class CenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.fc = nn.Linear(512, 2)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))


def load_model(model_path):
    model = CenterNet()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mark_image(image_np, center_norm, original_size, proc_time):
    h, w = original_size
    x = int(center_norm[0] * w)
    y = int(center_norm[1] * h)

    # Рисуем крест
    color = (0, 255, 0)  # Зеленый цвет
    thickness = 2
    size = 15
    cv2.line(image_np, (x - size, y), (x + size, y), color, thickness)
    cv2.line(image_np, (x, y - size), (x, y + size), color, thickness)

    # Добавляем текст с координатами и временем
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (0, 0, 0)  # Белый цвет
    text_thickness = 1

    # Текст координат
    coord_text = f"Center: ({x}, {y})"
    cv2.putText(image_np, coord_text, (x + size + 10, y),
                font, font_scale, text_color, text_thickness)

    # Текст времени обработки
    time_text = f"Time: {proc_time:.3f}s"
    cv2.putText(image_np, time_text, (10, 30),
                font, font_scale, text_color, text_thickness)

    # Текст даты и времени
    datetime_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image_np, datetime_text, (10, 60),
                font, font_scale, text_color, text_thickness)

    return image_np


def process_images(input_dir, output_dir, model):
    image_files = [f for f in Path(input_dir).glob("*") if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    total_images = len(image_files)

    if total_images == 0:
        print("Не найдено изображений для обработки!")
        return

    print(f"Найдено {total_images} изображений. Начало обработки...")

    # Инициализация трансформации
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    total_time = 0

    for img_path in tqdm(image_files, desc="Обработка"):
        try:
            start_time = time.time()

            # Загрузка изображения
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Ошибка загрузки: {img_path}")
                continue

            # Преобразование и предсказание
            with torch.no_grad():
                image_tensor = transform(image).unsqueeze(0)
                center = model(image_tensor).squeeze()
                center_np = center.detach().cpu().numpy()

            # Расчет времени обработки
            proc_time = time.time() - start_time
            total_time += proc_time

            # Разметка изображения
            marked_img = mark_image(
                image.copy(),
                center_np,
                image.shape[:2],
                proc_time
            )

            # Сохранение результата
            output_path = Path(output_dir) / f"marked_{img_path.name}"
            cv2.imwrite(str(output_path), marked_img)

        except Exception as e:
            print(f"Ошибка при обработке {img_path.name}: {str(e)}")

    # Итоговая статистика
    print("\nОбработка завершена!")
    print(f"Общее время: {total_time:.2f} сек")
    print(f"Обработано изображений: {len(image_files)}")
    if len(image_files) > 0:
        print(f"Среднее время на изображение: {total_time / len(image_files):.3f} сек")


if __name__ == "__main__":
    # Конфигурация
    INPUT_DIR = "training_dataset/cros/generate/output_dxf"
    OUTPUT_DIR = "output_marked2"
    MODEL_PATH = "best_model.pth"

    # Подготовка
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    try:
        # Загрузка модели
        print("Загрузка модели...")
        model = load_model(MODEL_PATH)

        # Обработка изображений
        process_images(INPUT_DIR, OUTPUT_DIR, model)

        print(f"\nРезультаты сохранены в: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")