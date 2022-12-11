import copy
from typing import Any, List, Dict
import uuid
import numpy as np


class MethodCallDescription:
    def __init__(self, method_name: str, inputs: List[int], outputs: List[int]):
        """Указание блоку вызвать метод с именем method_name на данных с блочных входов с номерами из списка inputs и
        положить вычисленный результат на блочные выходы с номерами из списка outputs"""
        self.method_name = method_name
        self.inputs = inputs
        self.outputs = outputs


# вход - это список (пример демодулятора data = [[info], [snr]]) выход - это список (пример блок АБГШ - [[data], [snr]])
# TODO класс должен быть абстрактным
class Block:

    def __init__(self):
        self.block_output = None
        self.id = uuid.uuid4()

    def execute(self, data: List, config: List[MethodCallDescription]) -> None:
        """
        data это данные со всех входов этого блока, т.е. это список списков.
        Далее пользователь сам настраивает какие методы блока будут вызваны, с каких входов будут взяты
        данные для работы методов и на какие выходы будут переданы результаты работы методов.
        """
        max_output_num = 0
        for method_descr in config:
            max_output_num = max(max_output_num, *method_descr.outputs)

        block_output = [[] for i in range(max_output_num + 1)]

        for method_descr in config:
            # как в метод вызываемый по имени передать входные аргументы?
            # с помощью * перед списком аргументов

            data_for_method = [data[i] for i in method_descr.inputs]
            method = getattr(self, method_descr.method_name)

            args = []
            for sub_list in data_for_method:
                args.append(sub_list[0])

            method_output = method(*args)

            # todo что происходит, когда выходов нет?
            for output_num in method_descr.outputs:
                block_output[output_num].append(method_output)

        self.block_output = copy.deepcopy(block_output)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)


class InformationBlock(Block):
    def __init__(self):
        super().__init__()
        # Далее нужно перечислить все важные параметры, возникающие во время моделирования
        self.ebn0_db = None

    def save_info(self, ebn0_db=None):
        """Этот метод вызывается моделью, чтобы сохранить в данном блоке все необходимые параметры моделирования,
        которые позже будут переданы в нужные блоки. Предполагается, что у метода может быть большое число аргументов."""
        self.ebn0_db = ebn0_db

    def get_ebn0_db(self):
        """Для каждого параметра нужен getter"""
        if self.ebn0_db is None:
            raise Exception("Параметр моделирования %s не был сохранен в информационном блоке" % 'ebn0_db')
        return self.ebn0_db


class Connection:
    def __init__(self, from_block: Block, from_output: int, to_block: Block, to_input: int):
        self.from_block = from_block
        self.from_output = from_output
        self.to_block = to_block
        self.to_input = to_input


# TODO все что ниже пока не надо. Это требует обсуждения . Мне кажется сначала надо разобраться с Block и  Model
class BlockModulator(Block):

    def __init__(self, bits_per_symbol, constellation: np.ndarray):
        super().__init__()
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation

    def process(self, data: np.ndarray) -> np.ndarray:
        pass


class BlockChannel(Block):

    def __init__(self, information_bits_per_symbol):
        super().__init__()
        self.information_bits_per_symbol = information_bits_per_symbol

    def process(self, data: np.ndarray, ebn0_db) -> np.ndarray:
        """Добавляем к сигналу необходимое кол-во шума"""
        pass

    def calc_noise_variance(self, ebn0_db):
        pass


class BlockDemodulator(Block):
    def __init__(self, bits_per_symbol, constellation: np.ndarray, mode='hard'):
        super().__init__()
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation
        self.mode = mode

    def process(self, data: np.ndarray, noise_variance=None) -> np.ndarray:
        """Демодулируем символы, возвращая биты в режиме 'hard' и llr-ы в режиме 'soft'.
        В мягком режиме требуется передать значение дисперсии шума"""
        pass


class BlockCoder(Block):
    def __init__(self):
        super().__init__()

    def get_code_rate(self) -> float:
        pass

    def encode(self, input_bits) -> Any:
        pass


class BlockConvolutionalCoder(BlockCoder):
    def __init__(self, trellis):
        super().__init__()
        self.trellis = trellis

    def get_code_rate(self) -> float:
        return self.trellis.k / self.trellis.n

    def encode(self, input_bits) -> Any:
        pass


class BlockDecoder(Block):

    def __init__(self, mode='hard'):
        super().__init__()
        self.mode = mode

    def decode(self, data) -> Any:
        pass


class BlockConvolutionalDecoder(BlockDecoder):
    def __init__(self, trellis, mode='hard'):
        super().__init__()
        self.trellis = trellis
        self.mode = mode

    @classmethod
    def from_coder(cls, coder: BlockConvolutionalCoder, mode='hard'):
        return cls(coder.trellis, mode)

    def decode(self, data) -> Any:
        pass


# class Coder:
#     def __init__(self):
#         pass
#
#     def get_code_rate(self) -> float:
#         pass
#
#     def encode(self, input_bits) -> Any:
#         pass
#
#
# class ConvolutionalCoder(Coder):
#
#     def __init__(self, memory=None, g_matrix=None, trellis: cc.Trellis = None):
#         """Конструктор либо получает готовый треллис, либо создает его. Треллис полностью описывает кодер"""
#         super().__init__()
#         if trellis is None:
#             if memory is not None and g_matrix is not None:
#                 self.trellis = cc.Trellis(memory, g_matrix)
#             else:
#                 raise ValueError("Для описания кодера нужен либо треллис, либо память и генераторная матрица "
#                                  "для построения треллиса")
#         else:
#             self.trellis = trellis
#
#     def get_code_rate(self) -> float:
#         return self.trellis.k / self.trellis.n
#
#     def encode(self, input_bits) -> Any:
#         return cc.conv_encode(input_bits, self.trellis)
#
#
# class Modulator:
#     def __init__(self, bits_per_symbol, constellation: np.ndarray):
#         """
#
#         :param bits_per_symbol: Количество битов в одном символе
#         :param constellation: Список символов в созвездии, представленных комплексными числами. При переводе позици
#         символа внутри списка в двоичную с.с. получим битовую последовательность, соответствующую данному символу
#         """
#         self.bits_per_symbol = bits_per_symbol
#         self.constellation = constellation
#         pass
#
#     def modulate(self, input_bits) -> Any:
#         pass
#
#
# class Channel:
#
#     def add_noise(self, symbols, ebn0_db, information_bits_per_symbol) -> Any:
#         pass
#
#     def calc_noise_variance(self, ebn0_db, information_bits_per_symbol) -> float:
#         pass
#
#
# class Demodulator:
#     def __init__(self, bits_per_symbol, constellation: np.ndarray, mode='hard'):
#         self.bits_per_symbol = bits_per_symbol
#         self.constellation = constellation
#         self.mode = mode
#
#     def demodulate_hard(self, symbols) -> Any:
#         pass
#
#     def demodulate_soft(self, symbols, noise_variance) -> Any:
#         pass
#
#
# class Decoder:
#     def decode(self, bits, llrs=None) -> Any:
#         pass
#
#
# class ConvolutionalDecoder(Decoder):
#     def __init__(self, trellis: cc.Trellis):
#         self.trellis = trellis
#
#     @classmethod
#     def from_coder(cls, coder: ConvolutionalCoder):
#         return cls(coder.trellis)
#
#     def decode(self, bits, llrs=None) -> Any:
#         if llrs is None:
#             return cc.viterbi_decode(bits, self.trellis)
#         else:
#             return cc.viterbi_decode(llrs, self.trellis, decoding_type='unquantized')
#         pass
