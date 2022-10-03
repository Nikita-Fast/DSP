import time

import numpy as np
import scipy.sparse as sps
from matplotlib import pyplot as plt


def bit_list_to_int(ls):
    """ Конвертирует битовую запись числа в его представление в десятичной системе счисления.
    Старшие биты находятся в начале списка."""
    l = len(ls)
    acc = 0
    for i in range(l):
        shift = l - 1 - i
        acc += ls[i] << shift
    return acc


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


class QAMModulator:
    """Класс описывающий КАМ модулятор"""

    def __init__(self, order):
        bits_per_symbol = np.log2(order)
        if bits_per_symbol != round(bits_per_symbol):
            raise ValueError("order must be 2 ** k, k = 2,4,6,...")

        self.modulation_order = order
        self.bits_per_symbol = int(bits_per_symbol)
        self.qam_symbols = self.create_qam_symbols()

    def __create_square_qam_symbols(self, order):
        """ Создаёт точки сигнального созвездия для квадратной M-QAM, где M это 4, 16, 64, ... ."""
        m = [i for i in range(order)]

        c = np.sqrt(order)
        b = -2 * (np.array(m) % c) + c - 1
        a = 2 * np.floor(np.array(m) / c) - c + 1
        s = list((a + 1j * b))
        return s

    def __create_cross_qam_symbols(self):
        """ Создаёт точки сигнального созвездия для крестовой M-QAM, где M это 32, 128, 512, ... ."""
        if (self.modulation_order == 2 or self.modulation_order == 8):
            raise ValueError(str(self.modulation_order) + "-QAM is not implemented yet")
        m = self.__create_square_qam_symbols(self.modulation_order // 2)
        l_row = int(np.sqrt(self.modulation_order // 2))
        n_row = int((self.modulation_order // 2) / (4 * l_row))

        other_points = np.empty(0, dtype=complex)
        # create left columns
        column = m[0:l_row]
        a = np.array(column)
        for i in range(n_row):
            b = a - 2 * (i + 1)
            other_points = np.append(other_points, b)

        # create right columns
        column = m[len(m) - l_row:len(m)]
        a = np.array(column)
        for i in range(n_row):
            b = a + 2 * (i + 1)
            other_points = np.append(other_points, b)

        # create top rows
        row = m[::l_row]
        a = np.array(row)
        for i in range(n_row):
            b = a + 2j * (i + 1)
            other_points = np.append(other_points, b)

        # create bottom rows
        row = m[l_row - 1::l_row]
        a = np.array(row)
        for i in range(n_row):
            b = a - 2j * (i + 1)
            other_points = np.append(other_points, b)

        other = list(other_points)
        return m + other

    def create_qam_symbols(self):
        """ Создаёт точки сигнального созвездия для КАМ модуляции."""
        if np.log2(self.modulation_order) % 2 == 0:
            m = [i for i in range(self.modulation_order)]

            c = np.sqrt(self.modulation_order)
            b = -2 * (np.array(m) % c) + c - 1
            a = 2 * np.floor(np.array(m) / c) - c + 1
            s = list((a + 1j * b))

            return np.array(s)
        else:
            return np.array(self.__create_cross_qam_symbols())

    def plot_constellation_points(self):
        """ Рисуем сигнальное созвездие."""
        points = self.qam_symbols
        plt.scatter(np.real(points), np.imag(points))
        plt.show()

    def modulate(self, bits):
        """ Преобразуем биты в КАМ символы"""
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError("Number of bits must be multiple of self.bits_per_symbol")

        ints = bits_to_ints(bits, self.bits_per_symbol)
        return self.qam_symbols[ints]
