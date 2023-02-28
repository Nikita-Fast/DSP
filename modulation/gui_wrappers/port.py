from typing import List
from uuid import uuid4

import PySide2
from PySide2.QtCore import Qt, QLineF
from PySide2.QtGui import QPen
from PySide2.QtWidgets import QGraphicsRectItem

from gui_wrappers.plain_connection import PlainConnection


class Port(QGraphicsRectItem):
    def __init__(self, parent, port_id, is_output_port):
        super().__init__(parent)
        self.parent = parent
        self.port_id = port_id
        self.uid = uuid4()

        self.height = self.get_default_height()
        h = self.height
        self.setRect(-h/2, -h/2, h, h)
        self.setBrush(Qt.red)
        self.setPen(QPen(Qt.black, 1))

        self.is_output_port = is_output_port

        # connections
        self.connections: List[PlainConnection] = []

    @staticmethod
    def get_default_height():
        return 50

    def __eq__(self, other):
        if isinstance(other, Port):
            return self.uid == other.uid
        return False

    def remove_all_connections(self):
        for c in self.connections:
            self.scene().removeItem(c)
        self.connections.clear()

    def update_connections(self):
        for c in self.connections:
            c.update_graphics()

    def mousePressEvent(self, event: PySide2.QtWidgets.QGraphicsSceneMouseEvent) -> None:
        print('%s-port-%i clicked! It has %i connections' % (self.parent.who_i_am(), self.port_id, len(self.connections)))
