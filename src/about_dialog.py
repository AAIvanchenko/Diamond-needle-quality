from PyQt5.QtWidgets import QDialog

from ui_about_dialog import UiAboutDialog


class AboutDialog(QDialog):
    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, *args, **kwargs)
        self.ui = UiAboutDialog()
        self.ui.setup_ui(self)
