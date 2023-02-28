from PySide2.QtCore import QLineF
from PySide2.QtWidgets import QGraphicsLineItem


class PlainConnection(QGraphicsLineItem):
    def __init__(self, line: QLineF, from_port=None, to_port=None, parent=None):
        super().__init__(line, parent)
        self.from_port = from_port
        self.to_port = to_port

    def update_graphics(self):
        new_line = QLineF(self.from_port.scenePos(), self.to_port.scenePos())
        self.setLine(new_line)