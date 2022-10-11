import numpy as np

import qam_modulation


class QAMDemodulator:
    """Класс описывающий КАМ демодулятор"""

    def __init__(self, bits_per_symbol: int, constellation_points):
        self.bits_per_symbol = bits_per_symbol
        self.constellation_points = constellation_points

    @classmethod
    def from_qam_modulator(cls, qam_modulator: qam_modulation.QAMModulator):
        return cls(bits_per_symbol=qam_modulator.bits_per_symbol,
                   constellation_points=qam_modulator.qam_symbols)

    def __ints_to_bits(self, ints):
        """Конвертирует массив int-ов в их битовое представление, за количество битов, выделяемых
        на каждый int отвечает поле bits_per_symbol."""
        b_len = self.bits_per_symbol
        if b_len > 16:
            raise Exception("Используется модуляция слишком высокого порядка. Поддерживаются только те, что "
                            "кодируют символ числом бит не превосходящим 16")
        if b_len > 8:
            bits = np.unpackbits(ints.astype(">u2").view("u1"))
            mask = np.tile(np.r_[np.zeros(16 - b_len, int), np.ones(b_len, int)], len(ints))
            return bits[np.nonzero(mask)]
        else:
            bits = np.unpackbits(ints.astype(np.uint8))
            mask = np.tile(np.r_[np.zeros(8 - b_len, int), np.ones(b_len, int)], len(ints))
            return bits[np.nonzero(mask)]

    def demodulate(self, symbols):
        """Демодулируем символы путём выбора точки созвездия до которой Эвклидово расстояние наименьшее.
        Вычисление происходит путём взятия модуля от матрицы, у которой столбцом является столбец переданных
        символов и из каждой строки вычли строку, составленную из точек сигнального созвездия.
        В целях экономии памяти обработка символов происходит группами"""
        c = np.array(self.constellation_points)

        l = len(symbols)
        idxs = [0]
        acc = 0

        magic_const = 51_200_000 // int(2 ** self.bits_per_symbol)
        while acc < l:
            acc = min(acc + magic_const, l)
            idxs.append(acc)

        n = len(idxs)
        z = zip(idxs[0:n - 1], idxs[1:n])
        pairs = [(i, j) for i, j in z]

        demod_ints = np.empty(l, dtype=int)
        for (a, b) in pairs:
            res = np.abs(symbols[a:b, None] - c[None, :]).argmin(axis=1)
            for i in range(a, b):
                demod_ints[i] = res[i - a]

        demod_bits = self.__ints_to_bits(demod_ints)
        return demod_bits
