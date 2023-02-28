import PySide2
import typing
from PySide2.QtCore import QSizeF, Qt, QLineF
from PySide2.QtWidgets import QGraphicsRectItem, QGraphicsItem, QGraphicsSimpleTextItem, QWidget, QLabel

from gui_wrappers.info_window import InfoWindow
from gui_wrappers.port import Port


class CBlock(QGraphicsRectItem):
    def __init__(self, num_input_ports, num_output_ports, parent: QGraphicsItem = None):
        super().__init__(parent)

        self.resize(200)
        flags = QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsScenePositionChanges
        self.setFlags(flags)

        self.setBrush(Qt.darkGray)

        # label
        self.label = QGraphicsSimpleTextItem(self.who_i_am(), self)
        self.label.setPen(Qt.NoPen)
        self.label.setAcceptHoverEvents(False)
        self.update_label_position()

        # ports
        self.height_between_ports = 20
        self.inputs = [Port(self, id, False) for id in range(num_input_ports)]
        self.outputs = [Port(self, id + num_input_ports, True) for id in range(num_output_ports)]
        self.update_ports_position()

        # info_window
        self.info_window = InfoWindow()
        self.info_window.set_text('')
        self.info_window.setWindowTitle(self.who_i_am())

    @staticmethod
    def who_i_am():
        return 'GUIBaseClass'

    def resize(self, size: float):
        self.setRect(-size / 2, -size / 2, size, size)

    def mousePressEvent(self, event: PySide2.QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            for p in self.inputs + self.outputs:
                p.remove_all_connections()
            self.scene().removeItem(self)

    def mouseDoubleClickEvent(self, event: PySide2.QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.info_window.set_text(self.who_i_am())
        self.info_window.show()

    def itemChange(self, change: PySide2.QtWidgets.QGraphicsItem.GraphicsItemChange, value: typing.Any) -> typing.Any:
        if change == QGraphicsItem.ItemScenePositionHasChanged and self.scene() is not None:
            for p in self.inputs + self.outputs:
                p.update_connections()
        return super().itemChange(change, value)

    def update_label_position(self):
        w = self.label.boundingRect().width()
        h = self.label.boundingRect().height()
        # center
        self.label.setPos(-w/2, -h/2)

    def update_ports_position(self):
        input_port_height_list, output_port_height_list = self.calc_ports_position()
        node_box = self.boundingRect()

        for i in range(len(self.inputs)):
            port = self.inputs[i]
            port.setX(-node_box.width()/2)
            port.setY(input_port_height_list[i])

        for i in range(len(self.outputs)):
            port = self.outputs[i]
            port.setX(node_box.width()/2)
            port.setY(output_port_height_list[i])

    def calc_ports_position(self):
        def resize_to_fit_ports(ports):
            num_ports = len(ports)
            port_height = Port.get_default_height()
            required_height = num_ports * port_height + (num_ports + 1) * self.height_between_ports
            if required_height > self.boundingRect().height():
                self.resize(required_height)

        def calc_height(num_ports, port_height):
            outside = self.boundingRect().height() - num_ports * port_height - (num_ports - 1) * self.height_between_ports
            h = self.boundingRect().height() / 2
            height = outside / 2 + port_height / 2 - h
            height_for_ports = []
            for i in range(num_ports):
                height_for_ports.append(height)
                height += (port_height + self.height_between_ports)
            return height_for_ports

        resize_to_fit_ports(self.inputs)
        resize_to_fit_ports(self.outputs)
        def_height = Port.get_default_height()

        return calc_height(len(self.inputs), def_height), calc_height(len(self.outputs), def_height)

