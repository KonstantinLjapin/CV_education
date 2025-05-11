import cv2
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path


def create_masks_from_images(
        input_dir: str,
        output_dir: str,
        threshold: int = 128,
        object_color: tuple = (255, 255, 255),
        mask_background: int = 0,
        mask_object: int = 255,
        dilation_kernel_size: int = 3
):
    """
    Создает бинарные маски из изображений PNG.

    Параметры:
    ----------
    input_dir : str
        Папка с исходными изображениями (PNG)
    output_dir : str
        Папка для сохранения масок
    threshold : int, optional
        Порог бинаризации (0-255), по умолчанию 128
    object_color : tuple, optional
        Цвет объекта для выделения (BGR), по умолчанию белый (255,255,255)
    mask_background : int, optional
        Значение фона в маске (0-255), по умолчанию 0
    mask_object : int, optional
        Значение объекта в маске (0-255), по умолчанию 255
    dilation_kernel_size : int, optional
        Размер ядра для морфологического расширения, по умолчанию 3
    """
    # Создаем папку для масок
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Получаем список PNG-файлов
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    print(f"Найдено {len(image_files)} изображений для обработки...")

    for img_file in tqdm(image_files, desc="Создание масок"):
        # Загружаем изображение
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)

        # Если изображение цветное, преобразуем в градации серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Вариант 1: Бинаризация по порогу
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Вариант 2: Выделение по цвету (если объект имеет определенный цвет)
        # lower = np.array([object_color[0]-10, object_color[1]-10, object_color[2]-10])
        # upper = np.array([object_color[0]+10, object_color[1]+10, object_color[2]+10])
        # mask = cv2.inRange(image, lower, upper)

        # Морфологические операции (опционально)
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Нормализуем значения маски
        processed[processed > 0] = mask_object
        processed[processed == 0] = mask_background

        # Сохраняем маску
        mask_filename = f"mask_{img_file}"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, processed)


if __name__ == "__main__":
    # Пример использования
    input_directory = "training_dataset/cros/generate/output_dxf"  # Замените на ваш путь
    output_directory = "masks/cross"  # Замените на ваш путь

    create_masks_from_images(
        input_dir=input_directory,
        output_dir=output_directory,
        threshold=200,  # Подберите под ваши изображения
        dilation_kernel_size=5
    )

    print("Генерация масок завершена!")