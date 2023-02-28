import PySide2
from PySide2.QtWidgets import QGraphicsRectItem, QWidget, QLabel

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
