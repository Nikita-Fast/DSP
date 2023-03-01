import PySide2
from PySide2.QtGui import Qt
from PySide2.QtWidgets import QGraphicsRectItem, QWidget, QLabel, QSlider, QSpinBox, QGridLayout

from channel import AWGNChannel
from gui_wrappers.block import CBlock


class ChannelGUI(CBlock):
    block: AWGNChannel

    def __init__(self, parent=None):
        super().__init__(2, 1, parent)
        self.block = AWGNChannel(2)

    @staticmethod
    def who_i_am():
        return 'AWGN Channel'

    def mouseDoubleClickEvent(self, event: PySide2.QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.interface = AWGNInterface()
        self.interface.setWindowTitle(self.who_i_am())
        self.interface.show()


class AWGNInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGeometry(400, 400, 400, 400)

        controls_layout = QGridLayout()

        self.label_bits_per_symbol = QLabel("bits_per_symbol:")
        self.spinbox_bits_per_symbol = QSpinBox(self)
        self.spinbox_bits_per_symbol.setRange(1, 12)
        self.spinbox_bits_per_symbol.setSingleStep(1)

        controls_layout.addWidget(self.label_bits_per_symbol, 0, 0)
        controls_layout.addWidget(self.spinbox_bits_per_symbol, 0, 1)

        self.setLayout(controls_layout)
