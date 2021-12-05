"""
Модуль, реализующий функции по обработки изображения.
Так же дополнительные функция для данной обработки.
"""

import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt
from typing import Callable


def draw_hist_gray(img_gray: np.ndarray) -> None:
    """
    Гистограмма распределения серого на картинке.

    :param img_gray: ЧБ изображение.
    """
    height, weight = img_gray.shape[:]
    pixel_sequence = img_gray.reshape([height * weight, ])
    number_bins = 256
    histogram, bins, patch = plt.hist(pixel_sequence, number_bins,
                                      facecolor="black", histtype="bar")
    plt.xlabel("Интенсивность серого")
    plt.ylabel("Количество пикселей")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()


def normalize_hist(img_gray: np.ndarray) -> np.ndarray:
    """
    Нормализация гистограммы распределения серого на картинке.

    :param img_gray: ЧБ изображение.

    :return: Нормализованное ЧБ изображение.
    """
    i_min, i_max = cv.minMaxLoc(img_gray)[:2]  # диапазон серого на картинке
    o_min, o_max = 0, 255  # диапазон уровней серого

    contrast = float(o_max - o_min) / (i_max - i_min)
    brightness = o_min - contrast * i_min

    out = (contrast * img_gray + brightness).astype(np.uint8)
    return out


def hist_equal(img_gray: np.ndarray) -> None:
    """
    Глобальное выравнивание гистограммы.

    Реализация выравнивания гистограммы в основном делится на четыре этапа:
        1.Рассчитайте гистограмму градаций серого изображения
        2.Рассчитать кумулятивную гистограмму серой гистограммы
        3.Соотношение между уровнем серого на входе и уровнем серого на выходе
        4.Циклически выводить уровень серого каждого пикселя изображения в
        соответствии с соотношением отображения
    В рамках данной функции использовалась встроенная в библиотеку
    OpenCV функция equalizeHist().

    :param img_gray: ЧБ изображение.
    """
    return cv.equalizeHist(img_gray)


def additive_correct(img: np.ndarray) -> np.ndarray:
    """
     Адаптивная коррекции гистограммы распределения серого на картинке.

     Адаптивное выравнивание гистограммы сначала делит изображение на
     непересекающиеся региональные блоки, а затем выполняет выравнивание
     гистограммы для каждого блока отдельно. Очевидно, что при отсутствии
     шума гистограмма в градациях серого для каждой небольшой области будет
     ограничена небольшим диапазоном градаций серого, но при наличии шума
     после выполнения выравнивания гистограммы для каждого блока разделенной
     области. Шум будет усиливаться.

     Во избежание появления шума предлагается «Ограничение контраста»
     (Contrast Limiting (clipLimit))

    :param img: Цветное или ЧБ изображение.

    :return: ЧБ изображение с адаптивной коррекцией гистограммы
    распределения серого.
    """
    shape_rgb_img = 3

    if len(img.shape) == shape_rgb_img:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    # Адаптивное пороговое выравнивание для ограничения контраста
    out = clahe.apply(img_gray)
    return out


def filter_strong_clarity(img: np.ndarray) -> np.ndarray:
    """
    Повышение резкости изображения при помощи определенного фильтра.
        Матрица фильтра выглядит следующим образом: [-1, -1, -1]
                                                    [-1, 9, -1]
                                                    [-1, -1, -1]
    :param: img: Цветная картинка.

    :return Цветная картинка с примененным фильтром.
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_filter = cv.filter2D(img, -1, kernel)
    return img_filter


def selective_filter_clarity(filter_clarity: Callable[[np.ndarray], np.ndarray], img: np.ndarray,
                             min_num_blur: int = 30) -> np.ndarray:
    """
    Выборочная фильтрацая изображений.
        У картинки будет повышаться резкость, если степень размыточти
        меньше заданного минимального значения.

    :param filter_clarity: Функция, которая повышает резкость изображения.
    :param img: Исходное цветное изображение или ЧБ изображение.
    :param min_num_blur: Минимальное число размыточти на картинке.

    :return: Цветное или ЧБ изображение с примененным фильтром.
    """
    num_blur = cv.Laplacian(img, cv.CV_64F).var()

    if num_blur < min_num_blur:
        filter_img = filter_clarity(img)
        return filter_img
    else:
        print("Фильтр не применился")
        return img


def filter_bilateral(img: np.ndarray) -> np.ndarray:
    """
    Фильтр размытия изображения - двусторонняя фильтрация.

    :param img: Исходное цветное изображение.

    :return: Цветное изображение с примененным фильтром.
    """
    diameter = 9  # Диаметр каждой окрестности пикселя, используемой во время фильтрации
    sigma_color = 75
    sigma_space = 75
    img_filter = cv.bilateralFilter(img, diameter, sigma_color, sigma_space)
    return img_filter


def filter_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """
    Фильтр размытия изображения - гауссовская фильтрация.

    :param img: Исходное цветное изображение.

    :return: Цветное изображение с примененным фильтром.
    """
    height_kernel, weight_kernel = 5, 5
    sigma_x = 0
    img_filter = cv.GaussianBlur(img, (height_kernel, weight_kernel), sigma_x)
    return img_filter


def filter_averag_img(img: np.ndarray) -> np.ndarray:
    """
    Фильтр размытия - усерднение изображения.

    :param img: Исходное цветное изображение.

    :return: Цветное изображение с примененным фильтром.
    """
    height_kernel, weight_kernel = 5, 5
    img_filter = cv.blur(img, (height_kernel, weight_kernel))
    return img_filter


def filter_add_weight(img: np.ndarray) -> np.ndarray:
    """
    Фильтр повышает резкость изображения.
        Вычисляет взвешенную сумму двух массивов.

    :param img: Цветная картинка.

    :return: Цветная картинка с примененным фильтром.
    """
    alpha = 4
    beta = -4
    gamma = 128

    kernel = cv.addWeighted(img, alpha, filter_gaussian_blur(img), beta, gamma)
    return kernel
