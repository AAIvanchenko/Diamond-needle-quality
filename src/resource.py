"""
Модуль, служащий преобразователем путей для статических файлов,
использующихся при сборке проекта в PyInstaller
"""

import os
import sys

def resource_path(relative_path):
    """
    Функция для преобразования путей статических файлов, использующихся
    совместно с PyInstaller.

    Функция служит для преобразования путей для статических файлов,
    использующихся при сборке проекта в PyInstaller.
    Если программа запущена из исходников, то вернётся относительный путь
    текущей директории.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
