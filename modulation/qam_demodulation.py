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

    def get_special_constellation_points(self, bit_value, bit_num):
        points = []
        for i in range(len(self.constellation_points)):
            mask = 1 << bit_num
            if i & mask == bit_value << bit_num:
                points.append(self.constellation_points[i])
        return points


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

    def demodulate(self, symbols, demod_type='hard', noise_var=1.0, use_formula=False):
        """Демодулируем символы путём выбора точки созвездия до которой Эвклидово расстояние наименьшее.
        Вычисление происходит путём взятия модуля от матрицы, у которой столбцом является столбец переданных
        символов и из каждой строки вычли строку, составленную из точек сигнального созвездия.
        В целях экономии памяти обработка символов происходит группами"""

        if demod_type == 'hard':
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
        elif demod_type == 'unquantized':
            llr_list = np.empty(self.bits_per_symbol * len(symbols))
            count = 0
            for r in symbols:
                if use_formula:
                    # method 1
                    # используем выведенную формулу
                    if self.bits_per_symbol == 2:
                        llr_b0 = 2*np.real(r)/noise_var
                        llr_b1 = -2*np.imag(r)/noise_var
                        llr_list[count] = llr_b1
                        llr_list[count+1] = llr_b0
                        count = count + 2
                    elif self.bits_per_symbol == 4:
                        var = -1/(2*noise_var)
                        re = np.real(r)
                        shifts = np.array([-3,-1,1,3])
                        e = np.exp(var * ((re + shifts)**2))
                        llr_b0 = np.log((e[0] + e[2]) / (e[1] + e[3]))
                        llr_b1 = np.log((e[0] + e[1]) / (e[2] + e[3]))

                        im = np.imag(r)
                        e = np.exp(var * (im + shifts)**2)
                        llr_b2 = np.log((e[1] + e[3]) / (e[0] + e[2]))
                        llr_b3 = np.log((e[2] + e[3]) / (e[0] + e[1]))

                        llr_list[count] = llr_b3
                        count = count + 1
                        llr_list[count] = llr_b2
                        count = count + 1
                        llr_list[count] = llr_b1
                        count = count + 1
                        llr_list[count] = llr_b0
                        count = count + 1
                    else:
                        raise ValueError("no LLR formula for QAM of order=%i" % (len(self.constellation_points)))
                else:
                    # method 2
                    # для вычисления LLR i-го бита считаем минимальное расстояние до точек созвездия сопоставленных
                    # с битовыми последовательностями, у которых на i-й позиции стоит бит 0. Анолично с битом равным 1.
                    # в качестве LLR выдаем разность двух минимальных расстояний
                    #начинаем со страших битов(левых)
                    for i in reversed(range(self.bits_per_symbol)):
                        d_bi_0 = float('inf')
                        for c in self.get_special_constellation_points(bit_value=0, bit_num=i):
                            dist = np.abs(r-c)
                            d_bi_0 = min(d_bi_0, dist)

                        d_bi_1 = float('inf')
                        for c in self.get_special_constellation_points(bit_value=1, bit_num=i):
                            dist = np.abs(r - c)
                            d_bi_1 = min(d_bi_1, dist)

                        llr_bi = d_bi_0 - d_bi_1
                        llr_list[count] = llr_bi
                        count = count + 1

            return llr_list
        else:
            raise ValueError("demod_type must be 'hard' or 'unquantized'")
