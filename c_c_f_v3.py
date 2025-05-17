import cv2
import numpy as np
import time


def detect_main_circle(image_path, blockSize=261, C=17, kernel_size=1, iterations=1):
    """Функция детекции главного круга"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Ошибка загрузки изображения {image_path}!")
        return None

    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize, C
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"Круг не обнаружен на изображении {image_path}!")
        return None

    main_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(main_contour)
    center = (int(x), int(y))
    radius = int(radius)

    result = np.zeros_like(img)
    cv2.circle(result, center, radius, 255, 2)
    return result


def process_image_with_timing(img_path):
    """Обработка одного изображения с замером времени"""
    timing_info = {}
    result_image = None
    circle_params = None

    try:
        # 1. Детекция круга
        start_time = time.time()
        circle_img = detect_main_circle(
            img_path,
            blockSize=261,
            C=17,
            kernel_size=1,
            iterations=1
        )
        timing_info['detection'] = time.time() - start_time

        if circle_img is None:
            return None, None, timing_info

        # 2. Поиск параметров круга
        contours, _ = cv2.findContours(circle_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        center = (int(x), int(y))
        circle_params = (center, radius)

        # 3. Загрузка исходника и рисование результата
        start_time = time.time()
        base_img = cv2.imread(img_path)
        if base_img is None:
            return None, None, timing_info

        result_image = base_img.copy()
        cv2.circle(result_image, center, int(radius), (0, 255, 0), 2)

        # Рисуем крестик
        cross_size = 20
        half = cross_size // 2
        cv2.line(result_image, (center[0] - half, center[1]), (center[0] + half, center[1]), (0, 0, 255), 2)
        cv2.line(result_image, (center[0], center[1] - half), (center[0], center[1] + half), (0, 0, 255), 2)

        timing_info['drawing'] = time.time() - start_time
        timing_info['total'] = sum(timing_info.values())

    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {str(e)}")
        return None, None, timing_info

    return result_image, circle_params, timing_info


if __name__ == "__main__":
    for i in range(1, 20):
        img_path = f"training_dataset/cros/reflection/{i}.jpg"

        print(f"\nОбработка изображения {i}.jpg...")
        result_img, params, timings = process_image_with_timing(img_path)

        if result_img is not None:
            print("Время выполнения:")
            for stage, t in timings.items():
                print(f"- {stage}: {t:.3f} сек")

            output_path = f"output_marked3/processed_{i}.jpg"
            cv2.imwrite(output_path, result_img)
            print(f"Результат сохранен как '{output_path}'")

            if params:
                center, radius = params
                print(f"Параметры круга:\n- Центр: {center}\n- Радиус: {radius}")