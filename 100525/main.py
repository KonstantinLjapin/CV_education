import asyncio
import time
import random
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw


def draw_box_random_centered(w, h, fc, mfs):
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    d0 = random.randint(0, min(w, h) // 2 - 1)
    if d0 < mfs // 2: d0 = mfs // 2 - 1
    x1 = w // 2 - d0
    y1 = h // 2 - d0
    x2 = w // 2 + d0
    y2 = h // 2 + d0

    draw.rectangle((x1, y1, x2, y2), fill=random.choice(fc), outline=(0, 0, 0))
    return img


def draw_circle_random_centered(w, h, fc, mfs):
    img = Image.new("RGB", (w, h), (255, 255, 255))  # Белый фон для наглядности
    draw = ImageDraw.Draw(img)

    # Максимально возможный радиус (чтобы круг не выходил за границы)
    max_r = min(w, h) // 2

    # Минимальный радиус (чтобы круг был видимым)
    min_r = mfs // 2
    if min_r < 1: min_r = 1  # Защита от нулевого радиуса

    # Случайный радиус в допустимых пределах
    r0 = random.randint(min_r, max_r)

    # Координаты квадрата, в который вписан круг
    x1 = w // 2 - r0
    y1 = h // 2 - r0
    x2 = w // 2 + r0
    y2 = h // 2 + r0

    # Рисуем круг (эллипс, вписанный в квадрат)
    draw.ellipse((x1, y1, x2, y2), fill=random.choice(fc), outline=(0, 0, 0))
    return img


_path = '../training_dataset'
if not os.path.exists(_path): os.makedirs(_path)
_folder = os.path.join(_path, "boxes")
if not os.path.exists(_folder): os.makedirs(_folder)
_folder = os.path.join(_path, "circles")
if not os.path.exists(_folder): os.makedirs(_folder)

num_samples = 1000
min_fig_size = 100  # Минимальный диаметр круга = 1000 (радиус = 500)
w = 1000  # Ширина изображения (должна быть >= min_fig_size)
h = 1000  # Высота изображения (должна быть >= min_fig_size)

# Список цветов в RGB формате
color_list = [
    (255, 0, 0),    # красный
    (0, 255, 0),    # зеленый
    (0, 0, 255),    # синий
    (255, 255, 0),  # желтый
    (128, 0, 128)   # фиолетовый
]


def generate_and_save(i):
    # Генерация и сохранение квадрата
    img_box = draw_box_random_centered(w=w, h=h, fc=color_list, mfs=min_fig_size)
    img_box.save(os.path.join(os.path.join(_path, "boxes"), f"box-{i}.png"))
    # Генерация и сохранение круга
    img_circle = draw_circle_random_centered(w=w, h=h, fc=color_list, mfs=min_fig_size)
    img_circle.save(os.path.join(os.path.join(_path, "circles"), f"circle-{i}.png"))


async def async_generate_and_save(i):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, generate_and_save, i)


async def main():
    tasks = [async_generate_and_save(i) for i in range(num_samples)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    print(f"Общее время выполнения: {time.time() - start_time:.2f} сек.")
    image_path = '../training_dataset/cros/original/26.PNG'
