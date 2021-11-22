import os
from platform import system


import numpy as np
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QSize


import contour
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

        self.image_directory: String = self.take_home_path()
        self.row_image: np.ndarray = None;
        self.ui_image: QImage = None;

        self.import_img.triggered.connect(self.load_image_by_dialog)

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
        file_path = QFileDialog.getOpenFileName(self,
                        "Загрузить изображение иглы",
                        self.image_directory,
                        "Image Files (*.png *.jpg)")[0]
        # Если пользователь не выбрал файл
        if file_path == "":
            return
        # Запоминание выбранной директории
        self.image_directory = "/".join(file_path.split("/")[:-1])
        self.row_image = contour.read_img(file_path)
        self.show_diamond_image()

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
            bytesPerLine = chanel * width
            self.ui_image = QImage(image.data, width, height,
                            bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        # Если чёрно-белая картинка
        elif len(image.shape) == 2:
            height, width = image.shape
            bytesPerLine = width
            self.ui_image = QImage(image.data, width, height,
                            bytesPerLine, QImage.Format_Mono)
        else:
            raise ValueError("Неверный размер массива изображеия")


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

    def show_diamond_image(self):
        """
        Метод, отвечающий за создание и отрисовку изображений.

        Сначала вызвает :method:'self.create_ui_image(image)' для
        создания изображение QImage из 'self.row_image'.
        После вызывает метод :method:'self.set_diamond_image_ui()' для
        отрисовки изображения в специальном виджете приложения.

        :ref:'self.create_ui_image()'
        :ref:'self.set_diamond_image_ui()'
        """
        self.create_ui_image(self.row_image)
        self.set_diamond_image_ui()

    def resizeEvent(self, event):
        """
        Метод, отлавливающий событие изменения размеров окна.

        При изменении размеров окна изменяет размер изображения
        иглы, вызывая :method:'self.set_diamond_image_ui()'

        :ref:'self.set_diamond_image_ui()'
        """
        if self.diamond_image.pixmap() is not None:
            self.set_diamond_image_ui()
