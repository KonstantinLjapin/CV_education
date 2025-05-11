import ezdxf
import os
import random
import math
import threading
from queue import Queue
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from concurrent.futures import ThreadPoolExecutor

# Глобальные параметры
PARAMS = {
    'input_dxf': "training_dataset/cros/dxf/cross1.dxf",
    'output_dir': "output_dxf",
    'num_images': 20,
    'base_scale': 30,
    'scale_variation': 0.05,  # ±5% variation
    'bg_colors': [(255, 255, 255), (240, 240, 240), (230, 230, 250)],
    'fill_colors': [(70, 130, 180), (220, 20, 60), (34, 139, 34), (218, 165, 32)],
    'line_colors': [(0, 0, 0), (50, 50, 50), (100, 100, 100)],
    'line_width_range': (2, 4),
    'max_rotation': 45,
    'threads': 4
}


def rotate_point(x, y, angle, cx, cy):
    """Поворот точки вокруг центра"""
    angle = math.radians(angle)
    x -= cx
    y -= cy
    new_x = x * math.cos(angle) - y * math.sin(angle)
    new_y = x * math.sin(angle) + y * math.cos(angle)
    return new_x + cx, new_y + cy


def random_scale(base_scale, variation):
    """Случайное масштабирование в пределах variation%"""
    return base_scale * (1 + random.uniform(-variation, variation))


def smart_fill(img, line_color, fill_color):
    """Умная заливка с сохранением контура"""
    contour = Image.new("L", img.size, 0)
    contour_draw = ImageDraw.Draw(contour)
    work_img = img.copy()
    pixels = work_img.load()

    line_r, line_g, line_b = line_color
    for x in range(img.width):
        for y in range(img.height):
            r, g, b = pixels[x, y]
            if abs(r - line_r) < 30 and abs(g - line_g) < 30 and abs(b - line_b) < 30:
                contour_draw.point((x, y), 255)

    contour = contour.filter(ImageFilter.MaxFilter(3))
    fill_mask = ImageOps.invert(contour)

    cx, cy = img.width // 2, img.height // 2
    if contour.getpixel((cx, cy)) == 255:
        found = False
        for r in range(1, min(img.width, img.height) // 2):
            for dx, dy in [(r, 0), (-r, 0), (0, r), (0, -r)]:
                x, y = cx + dx, cy + dy
                if 0 <= x < img.width and 0 <= y < img.height:
                    if contour.getpixel((x, y)) == 0:
                        cx, cy = x, y
                        found = True
                        break
            if found: break

    ImageDraw.floodfill(work_img, (cx, cy), fill_color, border=line_color)

    result = Image.new("RGB", img.size)
    result.paste(work_img)
    contour_pixels = contour.load()
    result_pixels = result.load()
    for x in range(img.width):
        for y in range(img.height):
            if contour_pixels[x, y] == 255:
                result_pixels[x, y] = line_color

    return result


def generate_image(task_queue):
    """Функция для потока, генерирующая изображения"""
    while not task_queue.empty():
        try:
            i, doc, msp, all_points, min_x, max_x, min_y, max_y = task_queue.get()

            # Случайные параметры для этого изображения
            angle = random.uniform(-PARAMS['max_rotation'], PARAMS['max_rotation'])
            scale = random_scale(PARAMS['base_scale'], PARAMS['scale_variation'])
            bg_color = random.choice(PARAMS['bg_colors'])
            fill_color = random.choice(PARAMS['fill_colors'])
            line_color = random.choice(PARAMS['line_colors'])
            line_width = random.randint(*PARAMS['line_width_range'])

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            size = max(max_x - min_x, max_y - min_y) * scale * 1.5
            img_size = int(size)

            img = Image.new("RGB", (img_size, img_size), bg_color)
            draw = ImageDraw.Draw(img)
            offset_x = img_size / 2 - (center_x - min_x) * scale
            offset_y = img_size / 2 - (center_y - min_y) * scale

            for entity in msp:
                if entity.dxftype() == "LINE":
                    x1, y1 = rotate_point(entity.dxf.start.x, entity.dxf.start.y, angle, center_x, center_y)
                    x2, y2 = rotate_point(entity.dxf.end.x, entity.dxf.end.y, angle, center_x, center_y)
                    start = ((x1 - min_x) * scale + offset_x, (max_y - y1) * scale + offset_y)
                    end = ((x2 - min_x) * scale + offset_x, (max_y - y2) * scale + offset_y)
                    draw.line([start, end], fill=line_color, width=line_width)
                elif entity.dxftype() in ("LWPOLYLINE", "POLYLINE"):
                    points = []
                    for vertex in entity.vertices():
                        x, y = rotate_point(vertex.dxf.location.x, vertex.dxf.location.y,
                                            angle, center_x, center_y)
                        px = (x - min_x) * scale + offset_x
                        py = (max_y - y) * scale + offset_y
                        points.append((px, py))
                    if len(points) > 2 and entity.closed:
                        draw.polygon(points, outline=line_color, fill=None, width=line_width)
                    elif len(points) > 1:
                        draw.line(points, fill=line_color, width=line_width)

            if fill_color != bg_color:
                img = smart_fill(img, line_color, fill_color)

            filename = os.path.join(PARAMS['output_dir'], f"cross_{i}.png")
            img.save(filename)
            print(f"Сгенерировано: {filename}")

            task_queue.task_done()
        except Exception as e:
            print(f"Ошибка в потоке: {e}")


def main():
    """Основная функция"""
    os.makedirs(PARAMS['output_dir'], exist_ok=True)

    try:
        doc = ezdxf.readfile(PARAMS['input_dxf'])
        msp = doc.modelspace()

        # Определение границ
        all_points = []
        for entity in msp:
            if entity.dxftype() == "LINE":
                all_points.append((entity.dxf.start.x, entity.dxf.start.y))
                all_points.append((entity.dxf.end.x, entity.dxf.end.y))
            elif entity.dxftype() in ("LWPOLYLINE", "POLYLINE"):
                for vertex in entity.vertices():
                    all_points.append((vertex.dxf.location.x, vertex.dxf.location.y))

        if not all_points:
            print("Не найдено ни одной точки в чертеже")
            return

        xs, ys = zip(*all_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Создаем очередь задач
        task_queue = Queue()
        for i in range(PARAMS['num_images']):
            task_queue.put((i, doc, msp, all_points, min_x, max_x, min_y, max_y))

        # Запускаем потоки
        with ThreadPoolExecutor(max_workers=PARAMS['threads']) as executor:
            for _ in range(PARAMS['threads']):
                executor.submit(generate_image, task_queue)

        print("Генерация завершена!")

    except Exception as e:
        print(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    main()
