import cv2
import numpy as np
import torch
import sys


class RobustCrossDetector:
    def __init__(self, template_path):
        try:
            self.template = torch.load(template_path)
            self.ref_points = self.template['points'].numpy()
            self.ref_center = self.template['center'].numpy()
            self.mean_color = self.template['mean_color'].numpy()
            self.img_size = self.template['image_size']
        except Exception as e:
            print(f"Ошибка загрузки шаблона: {str(e)}")
            sys.exit(1)

    def safe_contour_detection(self, img):
        """Надежное обнаружение контуров с обработкой ошибок"""
        try:
            # 1. Цветовая коррекция
            color_diff = cv2.absdiff(img, self.mean_color)
            gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)

            # 2. Адаптивная бинаризация
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 51, 5)

            # 3. Морфологическая обработка
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 4. Поиск контуров с проверкой
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # 5. Фильтрация по размеру
            min_area = self.img_size[0] * self.img_size[1] * 0.01  # 1% от площади изображения
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

            if not valid_contours:
                return None

            return valid_contours

        except Exception as e:
            print(f"Ошибка при обнаружении контуров: {str(e)}")
            return None

    def find_best_match(self, contours):
        """Нахождение лучшего совпадения с шаблоном"""
        try:
            best_match = None
            best_score = float('inf')

            for cnt in contours:
                # Аппроксимация контура
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Проверка на достаточное количество точек
                if len(approx) < 6:  # Минимум 6 точек для креста
                    continue

                # Сравнение форм через моменты
                match_score = cv2.matchShapes(approx, self.ref_points, cv2.CONTOURS_MATCH_I2, 0)

                if match_score < best_score:
                    best_score = match_score
                    best_match = cnt

            return best_match

        except Exception as e:
            print(f"Ошибка при сопоставлении контуров: {str(e)}")
            return None

    def detect(self, image_path):
        """Основная функция обнаружения"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Не удалось загрузить изображение")

            # 1. Обнаружение контуров
            contours = self.safe_contour_detection(img)
            if not contours:
                print("Не найдено подходящих контуров")
                return None

            # 2. Поиск лучшего совпадения
            best_cnt = self.find_best_match(contours)
            if best_cnt is None:
                print("Не найден контур, похожий на крест")
                return None

            # 3. Определение центра
            M = cv2.moments(best_cnt)
            if M["m00"] == 0:
                center = tuple(self.ref_center)
            else:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # 4. Визуализация
            result = img.copy()
            cv2.drawContours(result, [best_cnt], -1, (0, 255, 0), 3)

            # Рисуем перекрестие центра
            cross_size = max(img.shape[:2]) // 20
            cv2.line(result, (center[0] - cross_size, center[1]),
                     (center[0] + cross_size, center[1]), (0, 0, 255), 3)
            cv2.line(result, (center[0], center[1] - cross_size),
                     (center[0], center[1] + cross_size), (0, 0, 255), 3)

            # Вывод информации
            cv2.putText(result, f"Center: {center}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Detection Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return center

        except Exception as e:
            print(f"Ошибка при обнаружении креста: {str(e)}")
            return None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python detect.py <шаблон> <изображение>")
        sys.exit(1)

    try:
        detector = RobustCrossDetector(sys.argv[1])
        center = detector.detect(sys.argv[2])
        if center is not None:
            print(f"Найден центр креста: {center}")
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        sys.exit(1)