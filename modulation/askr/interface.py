import copy

from abc import ABC, abstractmethod
import uuid


"""
Здесь описаны только основные элементы модели. Частности не здесь быть должны
"""
class Port:
    """
    Это класс хранит данные и имеет уникальный номер. Его задача аккумулировать данные.
    Блок берет данные из порта или кладет в него. Здесь есть тонкий момент: предполагается что каждый блок
    на своем входе имеет заданный объем данных (или кратный чему-то).
    """
    def __init__(self):
        self.buffer = []
        self.id = uuid.uuid4()

    def put(self, data: list):
        self.buffer += data

    def get(self, num_elements):
        if len(self.buffer) < num_elements:
            return []
        output = self.buffer[:num_elements]
        self.buffer[:num_elements] = []
        return output

    def is_data_available(self):
        return len(self.buffer) > 0


class Block(ABC):
    """
    Блок имеет заданное количество портов входных и выходных
    Производный класс определяет их количество и определяет их назначение
    Производный класс сам забирает из них данные и кладет их на выход
    """
    def __init__(self, num_input_ports, num_output_ports):
        self.inputs = [Port()] * num_input_ports
        self.outputs = [Port()] * num_output_ports
        self.config = None
        self.input_buffer = None
        self.output_buffer = None
        self.id = uuid.uuid4()

    def execute(self):
        """
        Блоку все равно, что происходит вне него. Он производит обработку данных если данных на входе у него достаточно.
        Скорее всего необходимо сделать общий параметр для блока - размер шага. (а может и не надо - обсудим при встрече)
        """
        input_data_available = True
        for p in self.inputs:
            input_data_available = input_data_available and p.is_data_available()
        if input_data_available:
            self._execute_()

    @abstractmethod
    def _execute_(self):
        """
        Метод определяется производным классом и здесь описывается сама обработка (в том числе надо продумать обратную
        связь с gui для визуализации промежуточных результатов)
        """
        pass

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)


class Connection:
    def __init__(self, port_source: Port, port_destination: Port):
        self.source = port_source
        self.destination = port_destination
        self.id = uuid.uuid4()

    def flush(self):
        self.destination.put(self.source.get(-1))


class SourceBlock(Block, ABC):
    def __init__(self):
        super().__init__(0, 1)
