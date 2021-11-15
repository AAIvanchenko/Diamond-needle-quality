"""
Модуль, реализующий функцию построения прямых и расчёта статистик иглы.

:usecase: Наложение линий на иглу и получение её статистик:
    img = cv.imread(needle_imgs_path["path"])
    countur = // Функция вычисления котура
    statistic = build_line_and_find_statistic(img, countur)
"""

# Импортируем стандартные библиотеки
import math
from typing import Tuple
from collections.abc import Callable

# Импортируем сторонние библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.linear_model import LinearRegression


class Line:
    """
    Класс линии.

    В классе содержатся параметры линии (a, b), а также метод
    value для вычисления
    :math: 'y =  \\alpha x +  \\beta'.

    СЛУЖЕБНЫЙ

    :arg a: Параметр 'alpha' для линии из уравнения прямой
             :math: 'alpha =  \\alpha x +  \\beta'.
    :arg b: Параметр 'beta' для линии из уравнения прямой
             :math: 'y =  \\alpha x +  \\beta'.
    """
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def value(self, x: float) -> float:
        """
        Вычисление функции y = ax + b по x.

        :param x: Значение параметра х.

        :return: Вычисление функции y = ax + b.
        """
        return self.a * x + self.b

    def angle(self) -> float:
        """
        Вычисление угла между прямой и осью абсцисс (ОХ).

        :return: угол между прямой и осью абсцисс.
        """
        angle = math.degrees(math.atan(self.a))
        if angle < 0:
            angle = angle + 180
        return angle


class ContourLine:
    """
    Класс, содержащий линии, описывающие границы иглы.

    В классе содержатся 3 прямые, описывающие границы иглы, а также
    методы работы с ними.
    Стоит отметить, что на картинке левая и правая линия меняются
    местами, т.к. начало координат у картинки слева вверху.

    СЛУЖЕБНЫЙ

    :arg left: Кортеж, содержащий параметр 'alpha' и 'beta'
               для левой границы из уравнения прямой
               :math: 'y =  \\alpha x +  \\beta'.
    :arg right: Кортеж, содержащий параметр 'alpha' и 'beta'
               для правой границы из уравнения прямой
               :math: 'y =  \\alpha x +  \\beta'.
    :arg horizontal: Кортеж, содержащий параметр 'alpha' и 'beta'
               для горизонтальной границы из уравнения прямой
               :math: 'y =  \\alpha x +  \\beta'.
    """
    def __init__(self, left: Tuple[float, float],
                       right: Tuple[float, float],
                       horizontal: Tuple[float, float]):
        self.left_line = Line(*left)
        self.right_line = Line(*right)
        self.horizontal_line = Line(*horizontal)

    def __cross_point_lines(self, line1: Line, line2: Line) -> Tuple[float, float]:
        """
        Вычисление точки пересечения 2х прямых.

        :param line1: первая линия класса Line.
        :param line1: первая линия класса Line.

        :return: Кортеж (x, y), являющуюся координатами точки из
                 уравнения :math: 'y =  \\alpha x +  \\beta'.
        """
        numerator = line2.b - line1.b
        denumerator = line1.a - line2.a

        x = numerator / denumerator
        y = line1.value(x)
        return (x, y)

    def sharpening_angle(self) -> float:
        """
        Вычисление угла между прямыми левой и правой границ.

        :return: Угол между прямыми левой и правой границ в градусах.
        """
        alpha = self.right_line.angle() - self.left_line.angle()
        return round(alpha, 2)

    def area_triangle(self) -> float:
        """
        Вычисление площади треугольника, образованном 3мя прямыми.

        :return: Площадь треугольника, образованном 3мя прямыми.
        """
        # Точки пересечения прямых
        left_cross_point = np.array(self.__cross_point_lines(self.left_line, self.horizontal_line))
        middle_cross_point = np.array(self.__cross_point_lines(self.left_line, self.right_line))
        right_cross_point = np.array(self.__cross_point_lines(self.right_line, self.horizontal_line))
        # S = 1/2 ab * sin(apha)
        a = np.linalg.norm(left_cross_point - middle_cross_point)
        b = np.linalg.norm(right_cross_point - middle_cross_point)
        alpha = self.sharpening_angle()
        area = 0.5 * a * b * math.sin(math.radians(alpha))
        return round(area, 2)

    def tip_perpendicular_length(self) -> float:
        """
        Вычисление расстояние от точки пересечения левой и правой линии
        до горизонтальной линии.
        Расстояние вычисляется по формуле:
        :math: \frac{\lvert A x_0 + B y_0 + C\rvert}{\sqrt{ A^2 + B^2}},
            A - :math:'\alpha' горизонтальной прямой;
            B - 1;
            C - :math:'\beta' горизонтальной прямой;
            :math:'x_0' и :math:'y_0' - точка пересечения левой и правой
                линии.

        :return: Расстояние от точки пересечения левой и правой линии
                 до горизонтальной линии.
        """
        a = self.horizontal_line.a
        c = self.horizontal_line.b
        # Точка наконечника
        x, y = self.__cross_point_lines(self.left_line, self.right_line)
        numerator = abs(a * x - 1 * y + c)
        denumerator = math.sqrt(a**2 + 1**2)
        return round(numerator/denumerator, 2)



class LinearInterpolate:
    """
    Класс для построение модели ленейной интерполяции.

    При создании класса происходит создание модели LinearRegression.
    Класс повторяет основные необходимые методы модели LinearRegression:
    fit, predict и get_weight.

    СЛУЖЕБНЫЙ

    :arg df: Точки для обучения модели в формате DataFrame
             с колонками ["x", "y"].
    """
    def __init__(self, df: pd.DataFrame):
        self.model = LinearRegression(normalize = True, n_jobs=-1)
        self.fit(df["x"].to_numpy(), df["y"].to_numpy())

    def fit(self, x_list: list, y_list: list,
            loss_zone: float = 0.6, loss_ratio: int = 4):
        """
        Обучение модели ленейной интерполяции.

        :param x_list: массив точек признаков.
        :param y_list: массив значений таргетов.
        :param loss_zone: Параметр, отражающий размер зоны забвения в
                          процентах. Т.е. насколько далеко от края
                          изображения точки начнут отсеиваться.
                          Находится в пределах от 0.1 до 0.8.
        :param loss_ratio: Параметр, отражающий силу отсеиваться точек
                           в зоне забвения.
                           Задаётся целым числом от 3 до 20.
        """
        # Откинем лишние точки по мере их удалённости от наконечкика
        l = len(x_list)
        y_list_new = []
        x_list_new = []
        # Определяем коэфициенты
        if y_list[0] < y_list[-1]: # если левая граница
            sep_positions = [round(l*(loss_zone/2)), round(l*loss_zone)]
            loss_ratios = [loss_ratio, math.ceil(loss_ratio/2), 1]

        else:                      # если правая граница
            sep_positions = [round(l*(1-loss_zone)), round(l*(1-loss_zone/2))]
            loss_ratios = [1, math.ceil(loss_ratio/2), loss_ratio]
        # Производим отсеивание
        y_list_new = np.hstack([y_list[:sep_positions[0]:loss_ratios[0]],
            y_list[sep_positions[0]:sep_positions[1]:loss_ratios[1]],
            y_list[sep_positions[1]::loss_ratios[2]]])
        x_list_new = np.hstack([x_list[:sep_positions[0]:loss_ratios[0]],
            x_list[sep_positions[0]:sep_positions[1]:loss_ratios[1]],
            x_list[sep_positions[1]::loss_ratios[2]]])
        # Обучаем модель
        self.model.fit(x_list_new.reshape(-1, 1), y_list_new.reshape(-1, 1))

    def predict(self, x_list: np.ndarray) -> np.ndarray:
        """
        Возврат значений точек, вычисленных моделью ленейной интерполяции.

        :param x_list: массив точек признаков.

        :return: массив значений таргетов.
        """
        return self.model.predict(x_list.reshape(-1, 1))

    def get_weight(self) -> np.ndarray:
        """
        Получение весовых коэфициентов ленейной интерполяции.

        :return: массив весовых коэфициентов.
        """
        weight = self.model.coef_
        weight = np.append(weight, self.model.intercept_)
        return weight


class Statistic:
    """
    Класс для хранения статистик иглы.

    Хранит в себе статистики по изображению: угол заточки иглы и
    площадь стёртого наконечники в px^2.

    :arg angle: Угол между прямыми.
    :arg area_triangle: Полщадь недостающего наконечника.
    :arg length_missing_tip: Длина недостающей (сточенной) части
                             наконечника.
    """
    def __init__(self, sharpening_angle: float,
                       area_triangle: float,
                       length_missing_tip: float):
        self.angle = sharpening_angle
        self.area_triangle = area_triangle
        self.length_missing_tip = length_missing_tip


def __build_linear(contour: pd.DataFrame,
                   percent_cut: float = 0.1) -> ContourLine:
    """
    Построение линий по точкам границы.

    Строит 2 линии, описывающие боковые границы, и одну линию,
    описывающую тупую поверхность. Все эти линии хранятся в объекте
    класса ContourLine.

    :param contour: Точки контура в формате DataFrame
                    с колонками ["x", "y"].
    :param percent_cut: Процент, на который обрежется наконечник иглы.
                        По умолчкию равен 0.1.

    :return: Объект класса ContourLine, содержащий найденные прямые.
    """
    needle = contour.copy()
    # Обрезаем нижние границы
    needle = needle[needle["y"] > needle["y"].max() -
                            ((needle["y"].max() - needle["y"].min()) / 3)]
    # найдём точку максимума
    needle_max = needle.loc[[needle["y"].idxmax()]].to_numpy()[0]
    # разобъём на 2 боковых грани
    needle_edges = needle[needle["y"] < needle_max[1] -
                        percent_cut * (needle_max[1] - needle["y"].min())]
    # разобъём на боковые грани
    needle_edge_1 = needle_edges[needle_edges["x"] < needle_max[0]]
    needle_edge_2 = needle_edges[needle_edges["x"] > needle_max[0]]
    # построим регрессии для боковых граней
    needle_edge_1_interpolate = LinearInterpolate(needle_edge_1)
    needle_edge_2_interpolate = LinearInterpolate(needle_edge_2)
    # вычленим из них параметры
    (a1, b1) = needle_edge_1_interpolate.get_weight()
    (a2, b2) = needle_edge_2_interpolate.get_weight()
    # возвращаем функции линий:
    return ContourLine((a1, b1), (a2, b2), (0, needle_max[1]))


def plot_needle(df_list_to_plot: list, picture: np.ndarray):
    """
    Вывод изображения иглы и построение на нём линий.

    Строит 2 линии, описывающие боковые границы, и одну линию,
    описывающую тупую поверхность. Построение происходит с помощью
    библиотеки matplotlib.

    :param df_list_to_plot: Массив, каждый эллемент которого представляет
                            собой координаты точек, принадлежащих линии,
                            в формате DataFrame с колонками ["x", "y"].
    :param picture: Изображение в формате DataFrame.
    """
    plt.figure(figsize=(30, 13))
    plt.imshow(picture)

    linewidth = 3
    color = ["Blue", "Blue", "Green"]
    if df_list_to_plot is not None:
        for df in df_list_to_plot:
            plt.plot(df["x"], df["y"], linewidth = linewidth,
            alpha=0.8, color=color[0], linestyle = "dashed")
            color.pop(0)

    plt.title('Игла с построенными линиями')
    plt.show()


def build_line_and_find_statistic(img: pd.ndarray,
                                  contour: pd.DataFrame) -> Statistic:
    """
    Вывод изображения иглы, построение на нём линий, возврат статистики.

    Строит 2 линии, описывающие боковые границы, и одну линию,
    описывающую тупую поверхность. Выводит данные статистики картинки (
    угол заточки иглы и площадь стёртого наконечники в px^2).
    Возвращает статистику картинки в виде объекта класса Statistic.

    :param img: Изображение в формате DataFrame.
    :param contour: Точки контура в формате DataFrame
                    с колонками ["x", "y"].

    :return: Объект класса Statistic, содержащий найденные статистики.
    """
    # Строим линии по контуру
    linear = __build_linear(contour)
    # Получаем координаты точек линий в формате DataFrame
    left = contour["x"][0]
    right = contour["x"][contour.shape[0] - 1 ]
    left_line = pd.DataFrame([left, right], columns=["x"])
    left_line["y"] = linear.left_line.value(left_line["x"].to_numpy())
    right_line = pd.DataFrame([left, right], columns=["x"])
    right_line["y"] = linear.right_line.value(right_line["x"].to_numpy())
    top_line = pd.DataFrame([left, right], columns=["x"])
    top_line["y"] = linear.horizontal_line.value(top_line["x"].to_numpy())

    plot_needle(df_list_to_plot = [left_line, right_line, top_line],
                      picture = img)

    angle = linear.sharpening_angle()
    print("Угол заточки:", angle)
    area = linear.area_triangle()
    print("Площадь тупости в px^2:", area)
    length_missing_tip = linear.tip_perpendicular_length()
    print("Длина тупости в px:", length_missing_tip())

    statistic = Statistic(angle, area, length_missing_tip)

    return statistic
