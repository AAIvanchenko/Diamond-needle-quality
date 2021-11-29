"""
Модуль, реализующий функции по поиску границ алмазной иглы.
В модуле представлены все функции, которые были написаны в
ходе исследований различных методов выделения границ.
"""

from typing import List

import cv2 as cv
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def read_path(name_path: str) -> List[str]:
    """
    Сохранение путей до картинок в таблицу.

    :param name_path: Ссылка на таблицу с путями до картинок.

    :return: Список путей до каждой картинки.
    """
    list_img = pd.read_csv(name_path, sep=";")
    list_path_img = list_img["path"]
    return list_path_img


def read_img(path_img: str) -> np.ndarray:
    """
    Чтение картинки для дальнейшей работы.

    :param path_img: Полный путь до картинки.

    :return: Пиксельное представление картинки в формате RGB.
    """
    img = cv.imread(path_img, 1)
    return img


def max_points(contours: List[np.ndarray],
               number_pixel_min: int = 70) -> List[List[int]]:
    """
    Поиск максимума y для одинаковых x.
        Проходясь по всему массиву контуров, сначала мы отбираем
    те контуры, которые имеют, как минимум 35 пикселей. Далее соединяем
    все найденные границы. И ищем для каждого x максимальный y,
    так как отсчёт происходить с левого верхнего угла. Таким образом мы
    находим только нижнии точки границы.

    :param contours: Все найденные контуры алмазной иглы.
    :param number_pixel_min: Минимальное число пикселей в контуре

    :return: Список точек контура картинки.
    """
    filter_contours = [contour for contour in contours if contour.size
                       > number_pixel_min]
    contours_all = np.concatenate(filter_contours)
    contour_sort = sorted([x[0].tolist() for x in contours_all])
    x, y = contour_sort[0][0], contour_sort[0][1]
    points = []

    for point in contour_sort[1:]:
        if point[0] == x:
            if point[1] > y:
                y = point[1]
        else:
            points.append([x, y])
            x, y = point[0], point[1]

    return points


def find_contour(filter_img: np.ndarray) -> List[np.ndarray]:
    """
    Поиск контуров на картинки с примененным фильтром.

    :param filter_img: Изображение с выделенными границами.
    Псоле применения одного из фильтров.

    :return: Список контуров изображения.
    """
    threshold = 150
    max_val = 255
    _, thresh = cv.threshold(filter_img, threshold, max_val,
                             cv.THRESH_BINARY)
    contours, _ = cv.findContours(image=thresh, mode=cv.RETR_TREE,
                                  method=cv.CHAIN_APPROX_NONE)

    return contours


def filter_canny(img: np.ndarray) -> np.ndarray:
    """
    Поиск границ на картинке методом Кэнни.

    :param img: Цветная или ЧБ картинки.

    :return: Изображение с выделенными границами.
    """
    shape_rgb_img = 3

    if len(img.shape) == shape_rgb_img:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img

    threshold_one_canny = 100
    threshold_two_canny = 200
    filter_img = cv.Canny(gray_img, threshold_one_canny, threshold_two_canny)

    return filter_img


def filter_sobel(img: np.ndarray) -> np.ndarray:
    """
    Поиск границ на картинке методом Собеля.

    :param img: Цветная или ЧБ картинки.

    :return: Изображение с выделенными границами.
    """
    shape_rgb_img = 3
    if len(img.shape) == shape_rgb_img:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img

    x = cv.Sobel(gray_img, cv.CV_16S, 1, 0)
    y = cv.Sobel(gray_img, cv.CV_16S, 0, 1)

    # Конвертирование обратно в uint8
    abs_x = cv.convertScaleAbs(x)
    abs_y = cv.convertScaleAbs(y)

    # Приближаем градиент, добавив оба градиента направления
    # (обратите внимание, что это совсем не точный расчет)
    alpha = 0.5
    beta = 0.5
    gamma = 0
    filter_img = cv.addWeighted(abs_x, alpha, abs_y, beta, gamma)

    return filter_img


def filter_prewitt(img: np.ndarray) -> np.ndarray:
    """
    Поиск границ на картинке методом Превитта.
        Для выделение границ применяются матрицы по оси X b Y.

    :param img: Цветная или ЧБ картинки.

    :return: Изображение с выделенными границами.
    """
    shape_rgb_img = 3
    if len(img.shape) == shape_rgb_img:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img

    x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    img_x = cv.filter2D(gray_img, -1, x)
    img_y = cv.filter2D(gray_img, -1, y)

    filter_img = img_x + img_y

    return filter_img


def filter_roberts(img: np.ndarray) -> np.ndarray:
    """
    Поиск границ на картинке методом Робертса.
        Для выделение границ применяются матрицы по оси X b Y.

    :param img: Цветная или ЧБ картинки.

    :return: Изображение с выделенными границами.
    """
    shape_rgb_img = 3
    if len(img.shape) == shape_rgb_img:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img
    x = np.array([[1, 0], [0, -1]])
    y = np.array([[0, 1], [-1, 0]])

    img_x = cv.filter2D(gray_img, -1, x)
    img_y = cv.filter2D(gray_img, -1, y)

    filter_img = img_x + img_y

    return filter_img


def draw_borders(img: np.ndarray, point_contour: List[List[int]]) -> None:
    """
    Границы на исходной картинке.

    :param img: Исходная картинка.
    :param point_contour: Список точек контура картинки.
    """
    plt.figure(figsize=(13, 13))
    image_copy = img.copy()
    max_len = cv.drawContours(image=image_copy,
                              contours=np.array([point_contour]),
                              contourIdx=-1, color=(0, 255, 0),
                              thickness=2, lineType=cv.LINE_AA)
    plt.subplot(122), plt.imshow(max_len)
    plt.title('Границы иглы')

    plt.show()
