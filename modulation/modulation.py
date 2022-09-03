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


class QAMModulator:
    """Класс описывающий КАМ модулятор"""

    def __init__(self, order):
        bits_per_symbol = np.log2(order)
        if bits_per_symbol != round(bits_per_symbol):
            raise ValueError("order must be 2 ** k, k = 2,4,6,...")

        self.modulation_order = order
        self.bits_per_symbol = int(bits_per_symbol)

    def __square_qam_symbols(self, order):
        """ Создаёт точки сигнального созвездия для квадратной M-QAM, где M это 4, 16, 64, ... ."""
        m = [i for i in range(order)]

        c = np.sqrt(order)
        b = -2 * (np.array(m) % c) + c - 1
        a = 2 * np.floor(np.array(m) / c) - c + 1
        s = list((a + 1j * b))
        return s

    def __cross_qam_symbols(self):
        """ Создаёт точки сигнального созвездия для крестовой M-QAM, где M это 32, 128, 512, ... ."""
        if (self.modulation_order == 2 or self.modulation_order == 8):
            raise ValueError(str(self.modulation_order) + "-QAM is not implemented yet")
        m = self.__square_qam_symbols(self.modulation_order // 2)
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

    def qam_symbols(self):
        """ Создаёт точки сигнального созвездия для КАМ модуляции."""
        if np.log2(self.modulation_order) % 2 == 0:
            m = [i for i in range(self.modulation_order)]

            c = np.sqrt(self.modulation_order)
            b = -2 * (np.array(m) % c) + c - 1
            a = 2 * np.floor(np.array(m) / c) - c + 1
            s = list((a + 1j * b))

            return s
        else:
            return self.__cross_qam_symbols()

    def __bits_to_coo_matrix(self, bits):
        """ По массиву бит строим матрицу, где строка соответствет символу, а единичка в столбце -
        значению символа в десятичной системе счисления. Поскольку в строке все элементы кроме одного нулевые,
        используем разряженную матрицу в COO представлении."""

        length = len(bits) // self.bits_per_symbol
        m = sps.lil_matrix((length, self.modulation_order), dtype=bool)
        j = 0
        for i in range(length):
            ls = bits[self.bits_per_symbol * i: self.bits_per_symbol * (i + 1)]
            v = bit_list_to_int(ls)

            m[j, v] = 1
            j += 1
        return m.tocoo()

    def __modulate_using_matrix(self, bits):
        """ Преобразуем биты в символы и создаём матрицу где строка соответствует символу, а единичка в столбце -
        значению символа в десятичной системе счисления. Умножаем матрицу на столбец из комплексных чисел являющихся
        точками сигнального созвездия и получаем столбец промодулированных символов.
        Преобразуем полученный столбец в строку"""

        m = self.__bits_to_coo_matrix(bits)

        c_vector = np.array(self.qam_symbols())[:, None]

        t = time.time()
        modulated = m.dot(c_vector)
        res = modulated.ravel()
        print("vector product: %s sec", (time.time() - t))

        return res

    def plot_constellation_points(self):
        """ Рисуем сигнальное созвездие."""
        points = self.qam_symbols()
        plt.scatter(np.real(points), np.imag(points))
        plt.show()

    def __modulate_naive(self, bits):
        """Преобразуем биты в КАМ символы с помощью словаря."""
        i, j = 0, 0
        l = len(bits)
        out = np.empty(l // self.bits_per_symbol, dtype=complex)
        qam_symbol_dict = {k: v for k,v in zip(range(self.modulation_order), self.qam_symbols())}
        while i < l:
            bs = bits[i:i + self.bits_per_symbol]
            out[j] = qam_symbol_dict[bit_list_to_int(bs)]
            j += 1
            i += self.bits_per_symbol
        return out

    def modulate(self, bits):
        """ Преобразуем биты в КАМ символы"""
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError("Number of bits must be multiple of self.bits_per_symbol")

        # return self.__modulate_using_matrix(bits)
        return self.__modulate_naive(bits)
