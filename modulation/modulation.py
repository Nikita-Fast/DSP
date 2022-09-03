import time

import numpy as np
import scipy.sparse as sps


def bit_list_to_int(ls):
    l = len(ls)
    acc = 0
    for i in range(l):
        shift = l-1-i
        acc += ls[i] << shift
    return acc


class QAMModulator:

    def __init__(self, order):
        bits_per_symbol = np.log2(order)
        if bits_per_symbol != round(bits_per_symbol):
            raise ValueError("order must be 2 ** k, k = 2,4,6,...")

        self.modulation_order = order
        self.bits_per_symbol = int(bits_per_symbol)

    # def gray_encoding(self, dec_in):
    #     """ Encodes values by Gray encoding rule.
    #     Parameters
    #     ----------
    #     dec_in : list of ints
    #         Input sequence of decimals to be encoded by Gray.
    #     Returns
    #     -------
    #     gray_out: list of ints
    #         Output encoded by Gray sequence.
    #     """
    #
    #     bin_seq = [np.binary_repr(d, width=self.bits_per_symbol) for d in dec_in]
    #     gray_out = []
    #     for bin_i in bin_seq:
    #         gray_vals = [str(int(bin_i[idx]) ^ int(bin_i[idx - 1]))
    #                      if idx != 0 else bin_i[0]
    #                      for idx in range(0, len(bin_i))]
    #         gray_i = "".join(gray_vals)
    #         gray_out.append(int(gray_i, 2))
    #     return gray_out

    def __qam_symbols(self):
        """ Creates M-QAM complex symbols."""
        m = [i for i in range(self.modulation_order)]

        c = np.sqrt(self.modulation_order)
        b = -2 * (np.array(m) % c) + c - 1
        a = 2 * np.floor(np.array(m) / c) - c + 1
        s = list((a + 1j * b))
        return s

    def __bits_to_lil_matrix(self, bits):
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError("Number of bits must be multiple of self.bits_per_symbol")

        length = len(bits) // self.bits_per_symbol
        m = sps.lil_matrix((length, self.modulation_order), dtype=bool)
        j = 0
        for i in range(length):
            ls = bits[self.bits_per_symbol*i : self.bits_per_symbol*(i+1)]
            v = bit_list_to_int(ls)

            m[j, v] = 1
            j += 1
        return m

    def modulate_using_matrix(self, bits):
        lil = self.__bits_to_lil_matrix(bits)
        m = lil.tocoo()

        c_vector = np.array(self.__qam_symbols())[:, None]

        t = time.time()
        res = m.dot(c_vector)
        print("vector product: %s sec", (time.time() - t))

        return res.ravel()
