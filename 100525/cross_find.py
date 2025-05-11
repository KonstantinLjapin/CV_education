import cv2
import numpy as np
import os
from math import sqrt

# Фикс для Wayland
os.environ['QT_QPA_PLATFORM'] = 'xcb'


class CrossAnnotator:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Не удалось загрузить изображение")

        self.clone = self.image.copy()
        self.points = []  # Точки, указанные пользователем
        self.contour = None
        self.roi = None  # Область поиска

        cv2.namedWindow("Cross Annotation Tool")
        cv2.setMouseCallback("Cross Annotation Tool", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        """Обработка кликов мыши"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"Точка {len(self.points)}: ({x}, {y})")

                if len(self.points) == 2:
                    self.process_cross()

            self.redraw()

    def process_cross(self):
        """Обработка креста по двум точкам"""
        # 1. Определяем ROI (область поиска)
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]

        # Вычисляем центр и размер области
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size = int(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.2)  # +20% от размера

        # 2. Создаем маску области поиска
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask,
                      (max(0, center_x - size), max(0, center_y - size)),
                      (min(self.image.shape[1], center_x + size),
                       min(self.image.shape[0], center_y + size)),
                      255, -1)

        # 3. Предварительная обработка в ROI
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 51, 5)

        # 4. Поиск контуров в ROI
        masked = cv2.bitwise_and(binary, binary, mask=mask)
        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Выбираем контур с максимальной площадью в ROI
            self.contour = max(contours, key=cv2.contourArea)

            # Вычисляем центр контура
            M = cv2.moments(self.contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.points.append((cx, cy))  # Добавляем центр как точку 3

    def redraw(self):
        """Перерисовка изображения с разметкой"""
        self.clone = self.image.copy()

        # Рисуем текущие точки
        for i, (x, y) in enumerate(self.points):
            color = (0, 255, 0) if i < 2 else (0, 0, 255)  # Зеленые для углов, красный для центра
            cv2.circle(self.clone, (x, y), 8, color, -1)
            cv2.putText(self.clone, f"{i + 1}", (x + 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Рисуем контур если найден
        if self.contour is not None:
            cv2.drawContours(self.clone, [self.contour], -1, (255, 0, 0), 3)

            # Рисуем перекрестие в центре
            cx, cy = self.points[2]
            size = 30
            cv2.line(self.clone, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2)
            cv2.line(self.clone, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2)

        cv2.imshow("Cross Annotation Tool", self.clone)

    def save_results(self):
        """Сохранение результатов"""
        if len(self.points) < 3:
            print("Недостаточно точек для сохранения!")
            return

        # 1. Сохраняем размеченное изображение
        output_img_path = os.path.splitext(image_path)[0] + "_annotated.jpg"
        cv2.imwrite(output_img_path, self.clone)

        # 2. Сохраняем обрезанный крест
        x, y, w, h = cv2.boundingRect(self.contour)
        margin = int(max(w, h) * 0.2)  # 20% отступ
        cropped = self.image[max(0, y - margin):min(self.image.shape[0], y + h + margin),
                  max(0, x - margin):min(self.image.shape[1], x + w + margin)]
        output_crop_path = os.path.splitext(image_path)[0] + "_cropped.jpg"
        cv2.imwrite(output_crop_path, cropped)

        # 3. Сохраняем координаты
        output_txt_path = os.path.splitext(image_path)[0] + "_coords.txt"
        with open(output_txt_path, "w") as f:
            f.write(f"Top-left point: {self.points[0]}\n")
            f.write(f"Bottom-right point: {self.points[1]}\n")
            f.write(f"Auto-detected center: {self.points[2]}\n")
            f.write(f"Contour points: {len(self.contour)}\n")

        print(f"Результаты сохранены в:\n{output_img_path}\n{output_crop_path}\n{output_txt_path}")

    def run(self):
        """Основной цикл"""
        print("Инструкция:")
        print("1. Кликните по верху левого плеча креста")
        print("2. Кликните по низу правого плеча креста")
        print("3. Нажмите 's' для сохранения")
        print("4. Нажмите 'q' для выхода")

        self.redraw()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                self.save_results()

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()


# Пример использования
if __name__ == "__main__":
    image_path = '../training_dataset/cros/original/26.PNG'  # Укажите ваш путь

    try:
        annotator = CrossAnnotator(image_path)
        annotator.run()
    except Exception as e:
        print(f"Ошибка: {e}")