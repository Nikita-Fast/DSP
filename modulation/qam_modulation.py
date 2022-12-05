import numpy as np
from typing import Optional
from matplotlib import pyplot as plt

import default_qam_constellations
from interface import BlockModulator


def shifting(bit_list):
    out = 0
    for bit in bit_list:
        out = (out << 1) | bit
    return out


def bits_to_ints(bits, bits_per_int):
    i = 0
    symbols = np.empty(len(bits) // bits_per_int, dtype=int)
    k = 0
    while i < len(bits):
        symbols[k] = shifting(bits[i:i + bits_per_int])
        i += bits_per_int
        k += 1
    return symbols


def gray_codes(bits_per_symbol: int):
    if bits_per_symbol % 2 != 0:
        raise Exception("Генерация кодов Грея для нечетного bits_per_symbol ещё не реализована")
    order = 2 ** bits_per_symbol
    codes = []
    for i in range(order):
        codes.append(i ^ (i >> 1))

    length = int(np.sqrt(order))
    for i in range(length):
        if i % 2 == 1:
            start = i * length
            end = (i + 1) * length
            codes[start:end] = codes[start:end][::-1]
    return codes


def sort_constellation_points(complex_numbers):
    return sorted(complex_numbers, key=lambda x: (-x.imag, x.real))


class QAMModulator(BlockModulator):
    """Класс описывающий КАМ модулятор"""

    def __init__(self, bits_per_symbol: int = 4, constellation=default_qam_constellations.get_qam_constellation[4]):
        super().__init__(bits_per_symbol, constellation)

    def process(self, data: np.ndarray) -> np.ndarray:
        """ Преобразуем биты в КАМ символы"""
        # добавим нулевых битов, чтобы их общее число было кратно количеству битов приходящихся на один символ
        if len(data) % self.bits_per_symbol != 0:
            diff = len(data) % self.bits_per_symbol
            r = self.bits_per_symbol - diff
            data = np.pad(data, (0, r), 'constant')

        ints = bits_to_ints(data, self.bits_per_symbol)
        return self.constellation[ints]

    # def modulate(self, bits):
    #     """ Преобразуем биты в КАМ символы"""
    #
    #     # добавим нулевых битов, чтобы их общее число было кратно количеству битов приходящихся на один символ
    #     if len(bits) % self.bits_per_symbol != 0:
    #         diff = len(bits) % self.bits_per_symbol
    #         r = self.bits_per_symbol - diff
    #         bits = np.pad(bits, (0, r), 'constant')
    #
    #     ints = bits_to_ints(bits, self.bits_per_symbol)
    #     return self.constellation[ints]
