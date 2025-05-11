import cv2
import numpy as np
import random
import math


def generate_irregular_cross(output_path, width=1000, height=1000,
                             main_thickness=80, cross_thickness=70,
                             length=400, noise_level=30):
    """Генератор креста с неровными линиями и регулируемой толщиной"""

    # Создаем черное изображение
    img = np.zeros((height, width, 3), dtype=np.uint8)
    center = (width // 2, height // 2)

    # Параметры неровностей для обеих линий
    wave_amplitude = main_thickness * 0.3
    wave_frequency = 0.03
    thickness_variation = main_thickness * 0.2

    def draw_wavy_line(img, start, end, base_thickness, color):
        """Рисует линию с неровностями и изменяющейся толщиной"""
        points = []
        steps = 200

        # Генерируем основную траекторию с волнами
        for i in range(steps + 1):
            t = i / steps
            # Неровности для обеих осей
            offset_x = wave_amplitude * math.sin(t * wave_frequency * 100 + 1)
            offset_y = wave_amplitude * math.cos(t * wave_frequency * 100)

            x = int(start[0] * (1 - t) + end[0] * t + offset_x)
            y = int(start[1] * (1 - t) + end[1] * t + offset_y)
            points.append((x, y))

        # Рисуем линию с переменной толщиной
        for i in range(len(points) - 1):
            current_thickness = int(base_thickness +
                                    random.uniform(-thickness_variation, thickness_variation))
            cv2.line(img, points[i], points[i + 1], color,
                     max(1, current_thickness),
                     lineType=cv2.LINE_AA)

    # Рисуем вертикальную линию (более толстую с неровностями)
    draw_wavy_line(
        img,
        (center[0], center[1] - length),
        (center[0], center[1] + length),
        main_thickness,
        (255, 255, 255)
    )

    # Рисуем горизонтальную линию (тоже с неровностями)
    draw_wavy_line(
        img,
        (center[0] - length, center[1]),
        (center[0] + length, center[1]),
        cross_thickness,
        (255, 255, 255)
    )

    # Добавляем закругленные соединения
    connection_radius = max(main_thickness, cross_thickness) // 2
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        cx = center[0] + dx * (length - connection_radius)
        cy = center[1] + dy * (length - connection_radius)
        cv2.circle(img, (cx, cy), connection_radius,
                   (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # Добавляем реалистичные дефекты
    noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Размываем для естественного вида
    img = cv2.GaussianBlur(img, (7, 7), 0.8)

    # Улучшаем контраст
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)

    # Находим и размечаем контур
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Морфологическая обработка
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Находим 12 ключевых точек
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Аппроксимируем сложный контур
        epsilon = 0.015 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)

        # Выбираем 12 точек равномерно
        if len(approx) > 12:
            step = max(1, len(approx) // 12)
            approx = approx[::step][:12]

        # Размечаем точки на изображении
        for i, point in enumerate(approx):
            x, y = point[0]
            cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (x + 20, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Сохраняем результат
    cv2.imwrite(output_path, img)
    print(f"Сгенерирован крест с параметрами: толщина {main_thickness}/{cross_thickness}px")
    return approx.squeeze().tolist() if contours else []


if __name__ == "__main__":
    # Генерация креста с разными толщинами линий
    points = generate_irregular_cross(
        "wide_irregular_cross.jpg",
        main_thickness=90,  # Толщина вертикальной линии
        cross_thickness=80,  # Толщина горизонтальной линии
        length=450,  # Длина линий
        noise_level=25  # Уровень шума
    )

    # Вывод координат
    print("Координаты 12 точек контура:")
    for i, (x, y) in enumerate(points):
        print(f"Точка {i + 1}: ({x:4d}, {y:4d})")

    # Показ результата
    cv2.imshow("Wavy Cross", cv2.imread("wide_irregular_cross.jpg"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
