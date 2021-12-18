import os
import tempfile
from typing import Type, Union

from PyQt5.QtGui import QImage
from fpdf import FPDF


class CustomPdf(FPDF):
    """
    Класс, для создания pdf файла.
    А также создания шапки файла и нижней части pdf.

    :arg text_step: Шаг (расстояние) между строками текста.
    :arg font_size: Размер шрифта.
    """
    def __init__(self):
        super().__init__()
        self.text_step = 10
        self.font_size = 20

    def header(self) -> None:
        """
        Задание стиля для шапки файла.
        """
        self.add_font('DejaVuSerif-Bold', '',
                      '..//font//DejaVuSerif-Bold.ttf',
                      uni=True)
        font = self.set_font('DejaVuSerif-Bold', '', self.font_size)
        self.cell(0, self.text_step,
                  "Отчет.".format(font, self.font_size),
                  ln=2, align="C")

    def footer(self) -> None:
        """
        Задание стиля для нижней части файла.
        """
        position_page_number = -10
        self.set_y(position_page_number)
        # Добавляем номер страницы
        page_number = str(self.page_no())
        self.cell(0, self.text_step,
                  page_number,
                  self.font_size, 0, align="C")


def save_ui_img(q_img: Union[QImage]) -> str:
    """
    Сохранение картинки - QImage во временную папку.

    :param q_img: Картинка в формате QImage.

    :return: Путь к картинке во временной папке.
    """
    path_img = tempfile.mkdtemp() + 'needle.jpg'
    q_img.save(path_img)
    return path_img


def font_text(pdf: Type[CustomPdf]) -> None:
    """
    Задание шрифта для pdf файла.

    :param pdf: Экземпляр класса для создания pdf файла.
    """
    font_size = 14
    pdf.add_font('DejaVuSerif', '',
                 '..//font//DejaVuSerif.ttf',
                 uni=True)
    pdf.set_font('DejaVuSerif', '', font_size)


def add_text_before_img(pdf: Type[CustomPdf]) -> None:
    """
    Добавление текста после картинки.

    :param pdf: Экземпляр класса для создания pdf файла.
    """
    indent_heading = 5
    text_height = 10
    pdf.ln(indent_heading)
    pdf.cell(0, text_height,
             "Исследование иглы",
             ln=1, align="C")


def add_image(pdf: Type[CustomPdf], image_path: str) -> None:
    """
    Добавление картинки в отчет pdf.

    :param pdf: Экземпляр класса для создания pdf файла.
    :param image_path: Путь до изображения
    """
    text_height = 10
    indent = 115
    pdf.image(image_path, x=55, y=60, w=90)
    pdf.ln(indent)  # ниже на 130
    pdf.cell(0, text_height,
             txt="Рисунок 1 - Границы иглы".format(image_path),
             ln=1,  align="C")
    os.remove(image_path)


def add_text_result(pdf: Type[CustomPdf], corner: float, verdict: str) -> None:
    """
    Добавление текста о результатах работы программы.

    :param pdf: Экземпляр класса для создания pdf файла.
    :param corner: Угол заточки иглы.
    :param verdict: Вывод об игле (острая или тупая)
    """
    text_height = 10

    pdf.set_y(-100)
    pdf.cell(0, text_height,
             "Угол заточки: " + str(corner),
             ln=1, align="L")

    pdf.ln(5)
    pdf.cell(0, text_height,
             "Вывод игла: " + verdict,
             ln=1, align="L")


def create_pdf(pdf_path: str, q_image: Union[QImage], corner: float, verdict: str) -> None:
    """
    Генерация отчета pdf со всей информацией об игле.

    :param pdf_path: Путь для сохранения отчета pdf.
    :param q_image: Картинка в формате QImage.
    :param corner: Угол заточки иглы.
    :param verdict: Вывод об игле (острая или тупая)
    """
    pdf = CustomPdf()
    # Создаем особое значение {nb}
    pdf.alias_nb_pages()
    pdf.add_page()
    font_text(pdf)
    add_text_before_img(pdf)
    add_image(pdf, save_ui_img(q_image))
    add_text_result(pdf,
                    corner,
                    verdict)
    pdf.output(pdf_path)
