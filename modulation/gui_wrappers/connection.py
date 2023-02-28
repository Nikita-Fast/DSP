import math

import PySide2
import typing
from PySide2.QtCore import Qt, QRect, QSizeF, QLineF, QPointF, QSize, QPoint
from PySide2.QtGui import QPen, QPolygonF
from PySide2.QtWidgets import QGraphicsLineItem, QGraphicsItem

from gui_wrappers.port import Port


class ConnectionGUI(QGraphicsLineItem):
    arrow_head: QPolygonF

    def __init__(self, from_port: Port, to_port: Port, parent: QGraphicsItem=None):
        super().__init__(parent)
        self.from_port = from_port
        self.to_port = to_port
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setPen(QPen(Qt.black, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        self.arrow_head = QPolygonF()

    def boundingRect(self) -> PySide2.QtCore.QRectF:
        extra = (self.pen().width() + 20) / 2.0
        self.line().p1().x()
        return QRect(
            QPoint(int(self.line().p1().x()), int(self.line().p1().y())),
            QSize(
                self.line().p2().x() - self.line().p1().x(),
                self.line().p2().y() - self.line().p2().y()
            )
        ).normalized().adjusted(-extra, -extra, extra, extra)

    def shape(self) -> PySide2.QtGui.QPainterPath:
        path = self.shape()
        path.addPolygon(self.arrow_head)
        return path

    def updatePosition(self):
        line = QLineF(self.mapFromItem(self.from_port, 0, 0), self.mapFromItem(self.to_port, 0, 0))
        self.setLine(line)

    def paint(self, painter:PySide2.QtGui.QPainter, option:PySide2.QtWidgets.QStyleOptionGraphicsItem, widget:typing.Optional[PySide2.QtWidgets.QWidget]=...) -> None:
        if self.from_port.collidesWithItem(self.to_port):
            return
        arrow_size = 20.0
        my_pen = QPen(Qt.black)
        painter.setPen(my_pen)
        painter.setBrush(Qt.black)

        center_line = QLineF(self.from_port.pos(), self.to_port.pos())
        p1 = self.to_port.rect().topLeft() + self.to_port.pos()
        intersection_point = QPointF()

        points = [self.to_port.rect().topRight(), self.to_port.rect().bottomRight(), self.to_port.rect().bottomLeft()]
        for p2 in points:
            p2 += self.to_port.pos()
            poly_line = QLineF(p1, p2)
            intersection_type = poly_line.intersects(center_line, intersection_point)
            if intersection_type == QLineF.BoundedIntersection:
                break
            p1 = p2

        self.setLine(QLineF(intersection_point, self.from_port.pos()))
        angle = math.atan2(-self.line().dy(), self.line().dx())

        arrow_p1 = self.line().p1() + QPointF(
            math.sin(angle + math.pi / 3) * arrow_size,
            math.cos(angle + math.pi / 3) * arrow_size
        )

        arrow_p2 = self.line().p1() + QPointF(
            math.sin(angle + math.pi - math.pi / 3) * arrow_size,
            math.cos(angle + math.pi - math.pi / 3) * arrow_size
        )

        self.arrow_head.clear()
        self.arrow_head.append(self.line().p1())
        self.arrow_head.append(arrow_p1)
        self.arrow_head.append(arrow_p2)

        painter.drawLine(self.line())
        painter.drawPolygon(self.arrow_head)
        if self.isSelected():
            painter.setPen(QPen(Qt.black, 1, Qt.DashLine))
            my_line = self.line()
            my_line.translate(0, 4.0)
            painter.drawLine(my_line)
            my_line.translate(0, -8.0)
            painter.drawLine(my_line)
