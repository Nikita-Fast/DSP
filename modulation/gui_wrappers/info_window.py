from PySide2.QtWidgets import QWidget, QLabel


class InfoWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text_label = QLabel('', self)
        self.setGeometry(300, 300, 400, 400)

    def set_text(self, text: str):
        self.text_label.setText(text)
