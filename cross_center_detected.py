import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt


class CenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.fc = nn.Linear(512, 2)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))


def main():
    # Конфигурация (теперь внутри функции main)
    INPUT_DIR = "training_dataset/cros/generate/output_dxf"
    OUTPUT_DIR = "output_marked2"
    MODEL_PATH = "best_model.pth"
    IMG_SIZE = 256
    SHOW_PREVIEW = True
    PREVIEW_DELAY = 1000

    # Проверка и создание директорий
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    def load_model(model_path):
        model = CenterNet().to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def predict_center(model, image_np):
        image_tensor = transform(image_np).unsqueeze(0).to(device)
        with torch.no_grad():
            center = model(image_tensor)
        return center.squeeze().cpu().numpy()

    def mark_image(image_np, center_norm, original_size):
        h, w = original_size
        x = int(center_norm[0] * w)
        y = int(center_norm[1] * h)

        color = (0, 255, 0)
        thickness = 2
        size = 15

        cv2.line(image_np, (x - size, y), (x + size, y), color, thickness)
        cv2.line(image_np, (x, y - size), (x, y + size), color, thickness)

        text = f"({x}, {y})"
        cv2.putText(image_np, text, (x + size + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        return image_np

    def process_images(input_dir, output_dir, model):
        image_files = [f for f in Path(input_dir).glob("*") if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]

        print(f"Найдено {len(image_files)} изображений для обработки")

        for img_path in tqdm(image_files, desc="Обработка"):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Не удалось загрузить изображение: {img_path}")
                    continue

                original_size = image.shape[:2]
                center = predict_center(model, image)
                marked_img = mark_image(image.copy(), center, original_size)

                if SHOW_PREVIEW:
                    cv2.imshow("Preview", marked_img)
                    key = cv2.waitKey(PREVIEW_DELAY)
                    if key == 27:
                        cv2.destroyAllWindows()
                        break

                output_path = Path(output_dir) / f"marked_{img_path.name}"
                cv2.imwrite(str(output_path), marked_img)

            except Exception as e:
                print(f"Ошибка при обработке {img_path.name}: {str(e)}")

    print("Загрузка модели...")
    model = load_model(MODEL_PATH)

    print("Начало обработки изображений...")
    process_images(INPUT_DIR, OUTPUT_DIR, model)

    print(f"\nГотово! Результаты сохранены в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()