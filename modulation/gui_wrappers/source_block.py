from PySide2.QtWidgets import QGraphicsItem
from gui_wrappers.block import CBlock


class SourceBlock(CBlock):
    def __init__(self, num_output_ports=1, parent: QGraphicsItem = None):
        super().__init__(0, num_output_ports, parent)

    @staticmethod
    def who_i_am():
        return 'SourceBlock'
