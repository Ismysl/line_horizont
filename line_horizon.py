import cv2
import numpy as np

# Для чтения видео с файла
video_file = 'output28.avi'
cap = cv2.VideoCapture(video_file)

# Получиние разрешения кадров
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Определение кодек и создание объекта VideoWriter. Результат будет сохранен в файле 'output.avi'.
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Кодек безопасный вариант
out = cv2.VideoWriter('output.avi', fourcc, 15, (frame_width, frame_height))

# Флаг для отслеживания нажатия клавиши 't'
key_t_pressed = False

while cap.isOpened():
    ret, image = cap.read()  # Чтение каждого кадра видео
    if not ret:
        break

    # Конвертация в пространство HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Диапазон для неба
    blue_sky_lower = np.array([90, 50, 50])  # Небо - синие оттенки
    blue_sky_upper = np.array([130, 255, 255])

    # Диапазон для желтого поля
    yellow_field_lower = np.array([15, 50, 50])  # Желтый поле - желтые оттенки
    yellow_field_upper = np.array([40, 255, 255])

    # Создание масок для неба и желтого поля
    sky_mask = cv2.inRange(image_hsv, blue_sky_lower, blue_sky_upper)
    field_mask = cv2.inRange(image_hsv, yellow_field_lower, yellow_field_upper)

    # Объединение масок
    combined_mask = cv2.bitwise_or(sky_mask, field_mask)

    # Применение маски к исходному изображению
    result = cv2.bitwise_and(image, image, mask=combined_mask)

    # Нахождение контуров на маске
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Выбор самого большого контура
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Проверка на наличие точки (0, 0) в контуре
        zero_point_found = False
        for point in largest_contour:
            if point[0][0] == 0 and point[0][1] == 0: 
                zero_point_found = True
                break

        # Получаем точки с наибольшим значением по оси X
        max_x = max(point[0][0] for point in largest_contour)
        min_x = min(point[0][0] for point in largest_contour)
        max_x_points = [point for point in largest_contour if point[0][0] == max_x]
        min_x_points = [point for point in largest_contour if point[0][0] == min_x]

        if zero_point_found == False: # Если точки (0;0) нет  в контуре, т.е. контур определен в нижней части изображения, то интересующая нас линия - это минимальные значения y в контуре
            right_point = min(max_x_points, key=lambda point: point[0][1])
            left_point = min(min_x_points, key=lambda point: point[0][1])
        else:
            right_point = max(max_x_points, key=lambda point: point[0][1])
            left_point = max(min_x_points, key=lambda point: point[0][1])

        #print(f"Extreme points: Left ({left_point}), Right ({right_point})")

        # Отрисовка контура на изображении
        cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    else:
        print("No contours found")
        continue

    # Запись обработанного кадра обратно в видео файл
    out.write(image)

    # Отображение результата
    cv2.imshow('Detected Lines', image)

    # Проверка на нажатие клавиши 't'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        key_t_pressed = True
        found_coordinates = []  # Список для хранения найденных координат

        # проходим по всем точкам от минимум x до максимум x
        for x in range(int(min_x), int(max_x) + 1):
            filtered_points = [point for point in largest_contour if point[0][0] == x] # Список точек для каждого конкретного значения x

            if filtered_points:
                max_y = max(point[0][1] for point in filtered_points) # Находим макисмальное и минимальное значение y
                min_y = min(point[0][1] for point in filtered_points)

                if zero_point_found == False: # Если нет точки (0;0) в контуре, то берем минимальные значения y
                    found_coordinates.append((int(x), int(min_y)))
                else: # Если есть точка (0;0) в контуре, то берем максимальное значения y
                    found_coordinates.append((int(x), int(max_y)))

        print(f"Extreme points: Left ({left_point}), Right ({right_point})") # Вывод крайней левой и крайней правой точек кривой
        print(found_coordinates) # Вывод всех точек кривой

    if key == ord('q'):  # Завершение цикла при нажатии 'q'
        break

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()