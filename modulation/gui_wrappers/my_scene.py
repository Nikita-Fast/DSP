import PySide2
from PySide2.QtCore import Qt, QLineF
from PySide2.QtWidgets import QGraphicsScene, QGraphicsItem

from gui_wrappers.plain_connection import PlainConnection
from gui_wrappers.port import Port


class MyScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.line: PlainConnection = None

    @staticmethod
    def is_port(item: QGraphicsItem):
        return item.__class__.__name__ == Port.__name__

    def mousePressEvent(self, event: PySide2.QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            items = self.items(event.scenePos())
            if len(items) != 0 and self.is_port(items[0]):
                self.line = PlainConnection(line=QLineF(event.scenePos(), event.scenePos()))
                self.addItem(self.line)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: PySide2.QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if self.line is not None:
            new_line = QLineF(self.line.line().p1(), event.scenePos())
            self.line.setLine(new_line)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: PySide2.QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if self.line is not None:
            from_items = self.items(self.line.line().p1())
            to_items = self.items(self.line.line().p2())

            if len(from_items) and from_items[0] == self.line:
                from_items.remove(self.line)
            if len(to_items) and to_items[0] == self.line:
                to_items.remove(self.line)

            self.removeItem(self.line)

            if len(from_items) > 0 and len(to_items) > 0 and self.is_port(from_items[0]) and self.is_port(to_items[0]):
                from_port = from_items[0]
                to_port = to_items[0]
                # пока можно соединять порты только разных блоков
                if from_port.parent != to_port.parent:
                    # данные идут от выхода ко входу
                    if from_port.is_output_port and not to_port.is_output_port:
                        # выбранные два порта не должны иметь соединение
                        if all([
                            (c.from_port != from_port or c.to_port != to_port) and (
                                    c.from_port != to_port or c.to_port != from_port) for c in from_port.connections]):
                            pc = PlainConnection(
                                QLineF(from_items[0].scenePos(), to_items[0].scenePos()),
                                from_items[0],
                                to_items[0]
                            )
                            from_items[0].connections.append(pc)
                            to_items[0].connections.append(pc)
                            self.addItem(pc)
        self.line = None
        super().mouseReleaseEvent(event)
