"""
Модуль, реализующий главное окно.
"""

import os
from platform import system


import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import QSize, Qt


import contour
import linear
import imageproc
from ui_mainwindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Класс, реализующий главное окно приложения.

    Содержит следующие методы:
        :method:'take_home_path()' для возвращения домашней директории
                пользователя;
        :method:'load_image_by_dialog()' для вызызова диалогового окна
                для загрузки изображения;
        :method:'create_ui_image(image)' для создания QImage по
                переданному сырому изображению;
        :method:'set_diamond_image_ui()' для отображения
                'self.ui_image' в виджете изобаржения на главном
                экране приложения;
        :method:'create_linear()' для создания 'self.linear' на основе
                загруженнной картинки, режимов повышения резкости и
                типов фильтрации;
        :method:'create_statistic()' для добавления действий при изменении
                размеров окна;
        :method:'set_ui_statistic()' для отображения значения 'self.statistic'
                в полях пользовательского
        интерфейса;
        :method:'update_ui_image()' для обновления QImage с заданными
                дополнительными отрисовками;
        :method:'resizeEvent()' для добавления действий при изменении
                размеров окна;
    """
    def __init__(self):
        """
        Инициализация класса.
        """
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Проверка качества алмазной иглы")

        self.image_directory: str = self.take_home_path()
        self.image_name_file: str = None
        self.row_image: np.ndarray = None
        self.ui_image: QImage = None
        self.contour: np.ndarray = None
        self.linear: linear.ContourLine = None
        self.statistic: linear.Statistic = None

        self.sharpness_type.addItem("Нет")
        self.sharpness_type.addItem("Автоматическое")
        self.sharpness_type.addItem("Слабое")
        self.sharpness_type.addItem("Среднее")
        self.sharpness_type.addItem("Сильное")
        self.sharpness_type.setCurrentIndex(0)

        self.border_selection_type.addItem("Кенни")
        self.border_selection_type.addItem("Собеля")
        self.border_selection_type.setCurrentIndex(0)

        self.needle_boundaries_check.setChecked(True)
        # self.needle_line_check.setChecked(True)

        self.import_img.triggered.connect(self.load_image_by_dialog)
        self.export_pdf.triggered.connect(self.export_result_pdf)
        self.export_jpg.triggered.connect(self.export_result_image)
        self.needle_boundaries_check.clicked.connect(self.update_ui_image)
        self.needle_line_check.clicked.connect(self.update_ui_image)
        self.sharpness_type.currentIndexChanged.connect(self.create_linear)
        self.border_selection_type.currentIndexChanged.connect(
                                                        self.create_linear)

    def take_home_path(self) -> str:
        """
        Метод, возвращающий домашнюю директорию пользователя.

        Для Windows директория берётся из переменной окружения HOMEPATH
        (в виде C:/User/<username>).
        """
        home_path = ""
        system_name = system()
        if system_name == "Windows":
            home_path = os.environ['HOMEPATH']
        return home_path

    def load_image_by_dialog(self):
        """
        Метод, вызывающий диалоговое окно для загрузки изображения.

        Вызвает стандартное диалоговое окно каталога.
        Старотовый каталог берётся из переменной 'self.image_directory'.
        Выбранное изображение сохраняется в 'self.row_image'.
        Изменяет 'self.image_directory' на директорию, в котором было
        изображение.
        """
        file_path = QFileDialog.getOpenFileName(
                        self, "Загрузить изображение иглы",
                        self.image_directory, "Image Files (*.png *.jpg)")[0]
        # Если пользователь не выбрал файл
        if not file_path:
            return
        # Запоминание выбранной директории
        self.image_directory = os.path.dirname(file_path)
        self.image_name_file = os.path.basename(file_path)
        self.row_image = contour.read_img(file_path)
        self.create_linear()


    def create_ui_image(self, image: np.ndarray):
        """
        Метод, создающий QImage по переданному сырому изображению.

        Создаёт QImage по переданному сырому изображению в формате
        np.ndarray и сохраняет его в 'self.ui_image'. Переданное
        изображение может быть как серым (размера (n, m)),
        так и цветное (размера (n, m, 3)).

        :param image: картинка в формате массива np.ndarray
                      размера (n, m) или (n, m, 3).
        """
        if len(image.shape) == 3:
            height, width, chanel = image.shape
            bytes_per_line = chanel * width
            self.ui_image = QImage(image.data, width, height,
                                   bytes_per_line,
                                   QImage.Format_RGB888).rgbSwapped()
        # Если чёрно-белая картинка
        elif len(image.shape) == 2:
            height, width = image.shape
            bytes_per_line = width
            self.ui_image = QImage(image.data, width, height,
                                   bytes_per_line, QImage.Format_Mono)

    def set_diamond_image_ui(self):
        """
        Метод, отображабщий 'self.ui_image' в виджете изобаржения.

        Отображает QImage из 'self.ui_image' в эллементе QLabel
        'self.diamond_image'. Подгоняет изображение под размер
        'self.image_widget'.
        """
        # Подгонка по высоте
        height = self.image_widget.height()
        width = self.ui_image.width() * (height / self.ui_image.height())
        # Подгонка по ширине
        if width > self.image_widget.width():
            height = height * (self.image_widget.width() / width)
            width = self.image_widget.width()
        qpixmap = QPixmap(self.ui_image).scaled(QSize(width, height))
        self.diamond_image.setPixmap(qpixmap)

    def create_linear(self):
        """
        Метод, создающий 'self.linear' на основе загруженнной картинки,
        режимов повышения резкости и типов фильтрации.

        Создаёт объект типа CounterLine в переменной 'self.linear'
        на основе загруженнной картинки, режимов повышения резкости и
        типов фильтрации.
        По окончанию вызывает :method:'self.create_statistic()' и
        :method:'self.update_ui_image()'.

        :ref:'self.create_statistic()'
        :ref:'self.update_ui_image()'
        """
        img = self.row_image.copy()


        msg = QMessageBox()
        msg.setWindowTitle("Ошибка")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        # Увеличиваем резкость
        try:
            if self.sharpness_type.currentIndex() == 1:
                # Автоматическое
                img = imageproc.selective_filter_clarity(
                            imageproc.additive_correct,
                            img)
            elif self.sharpness_type.currentIndex() == 2:
                # Слабое
                img = imageproc.filter_add_weight(img)
            elif self.sharpness_type.currentIndex() == 3:
                # Среднее
                img = imageproc.additive_correct(img)
            elif self.sharpness_type.currentIndex() == 4:
                # Сильное
                img = imageproc.filter_strong_clarity(img)

            # Выделяем границы
            if self.border_selection_type.currentIndex() == 0:
                # Кени
                img = contour.filter_canny(img)
            elif self.border_selection_type.currentIndex() == 1:
                # Собеля
                img = contour.filter_sobel(img)
        except Exception:
            msg.setText("При применении фильтров возникла непредвиденная "+
                        "ошибка.\n\nПожалуйста, попробуйте ещё раз или " +
                        "возьмите другое изображение, или измените " +
                        "выбранные фильтры.")
            msg.exec_()
            return

        try:
            # Генерируем список точек границы
            contour_needle = contour.find_contour(img)
            contour_needle_max_point = contour.max_points(contour_needle)
            df_points = pd.DataFrame(contour_needle_max_point,
                                     columns=["x", "y"])
            self.contour = contour_needle_max_point
            # Находим линии, описывающие границы
            # try:
            self.linear = linear.build_contourline(df_points)
        except Exception:
            msg.setText("C выбранными элементами фильтрации контуров не " +
                        "найдено.\n\nПожалуйста, попробуйте изменить " +
                        "выбранный фильтр выделения границ или повышения " +
                        "резкости.")
            msg.exec_()
            self.update_ui_image()
            return

        self.create_statistic()
        self.update_ui_image()

    def create_statistic(self):
        """
        Метод, создающий 'self.statistic' на основе 'self.linear'.

        Создаёт объект типа Statistic в переменной 'self.statistic'
        на основе 'self.linear'.
        По окончанию вызывает :method:'self.set_ui_statistic()'.

        :ref:'self.set_ui_statistic()'
        """
        # Находим статистику иглы
        angle = self.linear.sharpening_angle()
        # print("Угол заточки:", angle)
        area = self.linear.area_triangle()
        # print("Площадь тупости в px^2:", area)
        length_missing_tip = self.linear.tip_perpendicular_length()
        # print("Длина тупости в px:", length_missing_tip)
        self.statistic = linear.Statistic(angle, area, length_missing_tip)
        self.statistic.make_sharping_result(
                                self.linear.horizontal_line.value(0))
        # print("Острая ли игла?:", self.statistic.is_sharp_result)

        self.set_ui_statistic()

    def set_ui_statistic(self):
        """
        Метод, отображабщий значения 'self.statistic' в полях
        пользовательского интерфейса.

        Отображает Statistic из 'self.statistic' в полях пользовательского
        интерфейса.
        """
        self.sharpening_angle.setText(str(self.statistic.angle))
        if self.statistic.is_sharp_result:
            self.conclusion.setText("Острая")
        else:
            self.conclusion.setText("Сточенная")

    def update_ui_image(self):
        """
        Метод, обновляющий QImage с заданными дополнительными отрисовками.

        Обновляет QImage в 'self.ui_image' с заданными дополнительными
        отрисовками (контуры иглы, построенные линии).
        """
        self.create_ui_image(self.row_image)
        painter = QPainter()
        painter.begin(self.ui_image)
        # Отрисовываем контур
        if (self.needle_boundaries_check.isChecked() and
            self.contour is not None):
            color = QColor(Qt.cyan)
            color.setAlphaF(0.5)
            painter.setPen(QPen(color, 5))
            for point in self.contour:
                painter.drawPoint(point[0], point[1])
        # Отрисовываем дополнительные линии
        if self.needle_line_check.isChecked() and self.linear is not None:
            colors = [QColor(Qt.blue), QColor(Qt.blue), QColor(Qt.red)]
            for color in colors:
                color.setAlphaF(0.8)
            for line in self.linear.get_lines():
                painter.setPen(QPen(colors[0], 8, Qt.DashLine))
                height = self.ui_image.width()
                painter.drawLine(0, line.value(0), height, line.value(height))
                colors.pop(0)
        painter.end()

        self.set_diamond_image_ui()

    def export_result_image(self):
        """
        Метод, вызывающий '' для сохранения отоброжаемого изображения.

        Метод вызывает метод модуля '' для сохранения отоброжаемого
        изображения с заданными параметрами отображения.

        :ref:''
        """
        msg = QMessageBox()
        msg.setWindowTitle("Ошибка")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        if self.ui_image is None:
            msg.setText("Экспорт недоступен. Загрузите сначала изображение " +
                        "и настройте параметры отображения.")
            msg.exec_()
            return

        intended_name = os.path.splitext(self.image_name_file)[0] + \
                                                                "_contour.jpg"
        file_path = QFileDialog.getSaveFileName(
                        self, "Экспорт отображаемого изображения иглы",
                        os.path.join(self.image_directory, intended_name),
                        "Image Files (*.png *.jpg)")[0]

        ext = os.path.splitext(file_path)[1]
        if ext not in ('.jpg', '.png'):
            msg.setText("Указанное расширение файла не доступно.")
            msg.exec_()
            return

        # Код, который запускает
        print(file_path)
        pass

    def export_result_pdf(self):
        """
        Метод, вызывающий '' для генерации PDF отчёта о проделанной работе.

        Метод вызывает метод модуля '' для генерации PDF отчёта о проделанной
        работе на основе данных из 'self.statistic'.
        Сохраняет дату генерации отчёта, отображаемое изображения, угол
        заточки и вердикт о тупости иглы.

        :ref:''
        """
        msg = QMessageBox()
        msg.setWindowTitle("Ошибка")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        if self.ui_image is None:
            msg.setText("Экспорт недоступен. Загрузите сначала изображение " +
                        "и настройте параметры отображения.")
            msg.exec_()
            return

        if self.statistic is None:
            msg.setText("Результаты программы не получены.\n " +
                        "Убедитесь, что программа отработала корректно.")
            msg.exec_()
            return

        intended_name = os.path.splitext(self.image_name_file)[0] + \
                                                                "_report.pdf"
        file_path = QFileDialog.getSaveFileName(
                        self, "Экспорт отчёта по текущему изображению иглы",
                        os.path.join(self.image_directory, intended_name),
                        "PDF Files (*.pdf)")[0]

        ext = os.path.splitext(file_path)[1]
        if ext != ".pdf":
            msg.setText("Указанное расширение файла не доступно.")
            msg.exec_()
            return

        # Код, который запускает
        print(file_path)
        pass

    def resizeEvent(self, _):
        """
        Метод, отлавливающий событие изменения размеров окна.

        При изменении размеров окна изменяет размер изображения
        иглы, вызывая :method:'self.set_diamond_image_ui()'

        :ref:'self.set_diamond_image_ui()'
        """
        if self.diamond_image.pixmap() is not None:
            self.set_diamond_image_ui()
