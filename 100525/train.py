import cv2
import numpy as np
import torch
import sys


class RobustCrossAnnotator:
    def __init__(self, image_path):
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError("Не удалось загрузить изображение")

            self.points = []
            cv2.namedWindow("Cross Annotation")
            cv2.setMouseCallback("Cross Annotation", self.mouse_callback)
            self.destroyed = False

        except Exception as e:
            print(f"Ошибка инициализации: {str(e)}")
            sys.exit(1)

    def mouse_callback(self, event, x, y, flags, param):
        try:
            if event == cv2.EVENT_LBUTTONDOWN and not self.destroyed:
                if len(self.points) < 12:
                    self.points.append((x, y))
                    print(f"Точка {len(self.points)}: ({x}, {y})")

                    if len(self.points) == 12:
                        self.save_template()
                        self.cleanup()
        except Exception as e:
            print(f"Ошибка в обработчике мыши: {str(e)}")
            self.cleanup()

    def save_template(self):
        """Безопасное сохранение шаблона"""
        try:
            if len(self.points) != 12:
                raise ValueError("Нужно ровно 12 точек")

            # Создаем маску с проверкой
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            points_array = np.array(self.points, dtype=np.int32)

            if points_array.size == 0:
                raise ValueError("Нет точек для создания маски")

            cv2.fillPoly(mask, [points_array], 255)

            # Находим центр через моменты
            M = cv2.moments(mask)
            if M["m00"] == 0:
                center = (self.image.shape[1] // 2, self.image.shape[0] // 2)
            else:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Сохраняем с проверкой
            template = {
                'points': torch.tensor(self.points),
                'center': torch.tensor(center),
                'mean_color': torch.tensor(cv2.mean(self.image, mask=mask)[:3]),
                'image_size': self.image.shape
            }

            torch.save(template, 'cross_template.pt')
            print("Шаблон успешно сохранен в cross_template.pt")

            # Визуализация
            vis = self.image.copy()
            cv2.polylines(vis, [points_array], True, (0, 255, 0), 2)
            cv2.circle(vis, center, 8, (0, 0, 255), -1)
            cv2.imwrite("template_visualization.jpg", vis)

        except Exception as e:
            print(f"Ошибка при сохранении шаблона: {str(e)}")
            raise

    def cleanup(self):
        """Корректное освобождение ресурсов"""
        try:
            if not self.destroyed:
                cv2.destroyAllWindows()
                self.destroyed = True
        except:
            pass

    def run(self):
        try:
            print("Кликайте по 12 точкам контура креста по часовой стрелке")
            while len(self.points) < 12 and not self.destroyed:
                img = self.image.copy()
                for i, (x, y) in enumerate(self.points):
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(img, str(i + 1), (x + 10, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.imshow("Cross Annotation", img)
                key = cv2.waitKey(1)
                if key == 27:  # ESC для выхода
                    break

        except Exception as e:
            print(f"Ошибка в основном цикле: {str(e)}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python train.py <путь_к_изображению>")
        sys.exit(1)

    try:
        annotator = RobustCrossAnnotator(sys.argv[1])
        annotator.run()
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        sys.exit(1)