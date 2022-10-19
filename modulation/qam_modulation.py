import numpy as np
from typing import Optional
from matplotlib import pyplot as plt

import default_qam_constellations


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
            end = (i+1) * length
            codes[start:end] = codes[start:end][::-1]
    return codes


def sort_constellation_points(complex_numbers):
    return sorted(complex_numbers, key=lambda x: (-x.imag, x.real))


class QAMModulator:
    """Класс описывающий КАМ модулятор"""

    def __init__(self, bits_per_symbol: int, bit_mapping: Optional):
        self.bits_per_symbol = bits_per_symbol

        if bit_mapping is None:
            self.qam_symbols = default_qam_constellations.get_qam_symbols_with_default_order[bits_per_symbol]
        else:
            # задаём свою битовую раскладку
            if len(bit_mapping) != 2 ** bits_per_symbol:
                raise ValueError("bit_mapping is not correct")

            qam_symbols = np.empty(len(bit_mapping), dtype=complex)
            default_order = default_qam_constellations.get_qam_symbols_with_default_order[bits_per_symbol]

            for i, (v, symbol) in enumerate(zip(bit_mapping, default_order)):
                qam_symbols[v] = default_order[i]

            self.qam_symbols = qam_symbols

    def modulate(self, bits):
        """ Преобразуем биты в КАМ символы"""
        # if len(bits) % self.bits_per_symbol != 0:
        #     raise ValueError("Number of bits must be multiple of self.bits_per_symbol")

        if len(bits) % self.bits_per_symbol != 0:
            diff = len(bits) % self.bits_per_symbol
            r = self.bits_per_symbol - diff
            bits = np.pad(bits, (0, r), 'constant')

        ints = bits_to_ints(bits, self.bits_per_symbol)
        return self.qam_symbols[ints]
