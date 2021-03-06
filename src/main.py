"""
Точка входа.
"""
import sys

from PyQt5 import QtWidgets

from mainwindow import MainWindow

if __name__ == "__main__":
    """
    Точка входа.
    """

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())
