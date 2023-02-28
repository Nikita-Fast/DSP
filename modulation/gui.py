import inspect
from typing import Dict

import PySide2
from PySide2.QtWidgets import QWidget, QGridLayout, QGroupBox, QComboBox, QPushButton, QVBoxLayout, QGraphicsView

from gui_wrappers.block import CBlock
from gui_wrappers.my_scene import MyScene
from gui_wrappers import *


class MainWindow(QWidget):
    view: QGraphicsView
    scene: MyScene
    descr_to_gui_class_dict: Dict

    def __init__(self):
        super().__init__()
        self.button_create_block = None
        self.combo_box_block_type = None
        self.group_box = None
        self.init_ui()

    @staticmethod
    def load_gui_classes():
        import gui_wrappers
        classes = []
        for _, module in inspect.getmembers(gui_wrappers):
            if inspect.ismodule(module):
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        classes.append(obj)
        gui_classes = []
        for cls in classes:
            if CBlock in cls.__bases__:
                gui_classes.append(cls)

        return gui_classes

    def add_gui_classes_to_combo_box(self):
        for i, x in enumerate(self.load_gui_classes()):
            self.combo_box_block_type.insertItem(i, x.who_i_am())

    def create_description_to_gui_class_dict(self):
        self.descr_to_gui_class_dict = {}
        for x in self.load_gui_classes():
            self.descr_to_gui_class_dict[x.who_i_am()] = x

    def init_ui(self):
        self.setGeometry(0, 0, 2000, 2000)
        self.setWindowTitle('Opa pyside2')

        layout = QGridLayout()

        self.group_box = QGroupBox('Control panel', self)

        self.combo_box_block_type = QComboBox()
        self.button_create_block = QPushButton('Create block!', self)
        self.button_create_block.clicked.connect(self.create_block_on_btn_click)

        vbox = QVBoxLayout()
        vbox.addWidget(self.combo_box_block_type)
        vbox.addWidget(self.button_create_block)
        vbox.addStretch(1)

        self.group_box.setLayout(vbox)
        layout.addWidget(self.group_box, 0, 0)

        self.add_gui_classes_to_combo_box()
        self.create_description_to_gui_class_dict()

        self.create_graphic_view()
        layout.addWidget(self.view, 0, 1)

        self.setLayout(layout)
        self.show()

    def add_block(self, block_descr: str):
        block_class = self.descr_to_gui_class_dict[block_descr]
        block_obj = block_class()
        self.scene.addItem(block_obj)

    def create_graphic_view(self):
        self.scene = MyScene()
        self.view = QGraphicsView(self.scene, self)
        self.view.setGeometry(0, 0, 2000, 2000)

    def create_block_on_btn_click(self):
        curr_text = self.combo_box_block_type.currentText()
        self.add_block(curr_text)

    def mousePressEvent(self, event: PySide2.QtGui.QMouseEvent) -> None:
        """Закрываем окно с подробной информацией о блоке, если нажали мышью где-то на главном окне, но не попали
        в информационное окно"""
        # todo если нажать на кнопку, то информационные окно не закроются
        a = [x for x in self.scene.items() if CBlock in x.__class__.__bases__]
        for x in a:
            if x.info_window is not None:
                x.info_window.close()
