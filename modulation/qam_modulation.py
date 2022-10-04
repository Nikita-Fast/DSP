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
            k = 0
            for v in bit_mapping:
                qam_symbols[v] = default_order[k]
                k += 1
            self.qam_symbols = qam_symbols

    def modulate(self, bits):
        """ Преобразуем биты в КАМ символы"""
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError("Number of bits must be multiple of self.bits_per_symbol")

        ints = bits_to_ints(bits, self.bits_per_symbol)
        return self.qam_symbols[ints]

    # def create_square_qam_symbols(self, bits_per_symbol):
    #     """ Создаёт точки сигнального созвездия для квадратной M-QAM, где M это 4, 16, 64, ... ."""
    #     order = 2 ** bits_per_symbol
    #     m = [i for i in range(order)]
    #
    #     c = np.sqrt(order)
    #     b = -2 * (np.array(m) % c) + c - 1
    #     a = 2 * np.floor(np.array(m) / c) - c + 1
    #     s = list((a + 1j * b))
    #     return s

    # def create_cross_qam_symbols(self):
    #     """ Создаёт точки сигнального созвездия для крестовой M-QAM, где M это 32, 128, 512, ... ."""
    #     order = 2 ** self.bits_per_symbol
    #     if order == 2 or order == 8:
    #         raise ValueError(str(order) + "-QAM is not implemented yet")
    #
    #     m = self.create_square_qam_symbols(self.bits_per_symbol - 1)
    #     l_row = int(np.sqrt(order // 2))
    #     n_row = int((order // 2) / (4 * l_row))
    #
    #     other_points = np.empty(0, dtype=complex)
    #     # create left columns
    #     column = m[0:l_row]
    #     a = np.array(column)
    #     for i in range(n_row):
    #         b = a - 2 * (i + 1)
    #         other_points = np.append(other_points, b)
    #
    #     # create right columns
    #     column = m[len(m) - l_row:len(m)]
    #     a = np.array(column)
    #     for i in range(n_row):
    #         b = a + 2 * (i + 1)
    #         other_points = np.append(other_points, b)
    #
    #     # create top rows
    #     row = m[::l_row]
    #     a = np.array(row)
    #     for i in range(n_row):
    #         b = a + 2j * (i + 1)
    #         other_points = np.append(other_points, b)
    #
    #     # create bottom rows
    #     row = m[l_row - 1::l_row]
    #     a = np.array(row)
    #     for i in range(n_row):
    #         b = a - 2j * (i + 1)
    #         other_points = np.append(other_points, b)
    #
    #     other = list(other_points)
    #     return m + other
    #
    # def create_qam_symbols(self):
    #     """ Создаёт точки сигнального созвездия для КАМ модуляции."""
    #     order = 2 ** self.bits_per_symbol
    #     if np.log2(order) % 2 == 0:
    #         m = [i for i in range(order)]
    #
    #         c = np.sqrt(order)
    #         b = -2 * (np.array(m) % c) + c - 1
    #         a = 2 * np.floor(np.array(m) / c) - c + 1
    #         s = list((a + 1j * b))
    #
    #         return np.array(s)
    #     else:
    #         return np.array(self.create_cross_qam_symbols())
    #
    # def plot_constellation_points(self):
    #     """ Рисуем сигнальное созвездие."""
    #     points = self.qam_symbols
    #     plt.scatter(np.real(points), np.imag(points))
    #     plt.show()

