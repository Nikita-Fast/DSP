import time

import numpy as np

def distance(p1, p2):
    return np.abs(p1-p2)

class QAMDemodulator:

    def __init__(self, order):
        bits_per_symbol = np.log2(order)
        if bits_per_symbol != round(bits_per_symbol):
            raise ValueError("order must be 2 ** k, k = 2,4,6,...")

        self.modulation_order = order
        self.bits_per_symbol = int(bits_per_symbol)

    def __qam_symbols(self):
        """ Creates M-QAM complex symbols."""
        m = [i for i in range(self.modulation_order)]

        c = np.sqrt(self.modulation_order)
        b = -2 * (np.array(m) % c) + c - 1
        a = 2 * np.floor(np.array(m) / c) - c + 1
        s = list((a + 1j * b))
        return s

    def __ints_to_bits(self, ints):
        b_len = self.bits_per_symbol
        if b_len > 8:
            bits = np.unpackbits(ints.astype(">u2").view("u1"))
            # bits = np.unpackbits(ints.astype(np.uint16))
            mask = np.tile(np.r_[np.zeros(16-b_len, int), np.ones(b_len, int)], len(ints))
            return bits[np.nonzero(mask)]
        else:
            bits = np.unpackbits(ints.astype(np.uint8))
            mask = np.tile(np.r_[np.zeros(8 - b_len, int), np.ones(b_len, int)], len(ints))
            return bits[np.nonzero(mask)]


    def demodulate(self, symbols):
        c = np.array(self.__qam_symbols())

        l = len(symbols)
        idxs = [0]
        acc = 0
        while acc < l:
            acc = min(acc + 20_000, l)
            idxs.append(acc)

        n = len(idxs)
        z = zip(idxs[0:n - 1], idxs[1:n])
        pairs = [(i, j) for i, j in z]

        demod_ints = np.empty(l, dtype=int)
        for (a, b) in pairs:
            res = np.abs(symbols[a:b, None] - c[None, :]).argmin(axis=1)
            for i in range(a, b):
                demod_ints[i] = res[i-a]

        demod_bits = self.__ints_to_bits(demod_ints)
        return demod_bits


    def demodulate_naive(self, symbols):
        c = self.__qam_symbols()
        demod_ints = np.empty(len(symbols), dtype=int)
        j=0

        for i in range(len(symbols)):
            dists = list(map(lambda x: distance(symbols[i], x), c))
            m = min(dists)
            idx = dists.index(m)
            demod_ints[j] = idx
            j+=1

        demod_bits = self.__ints_to_bits(demod_ints)
        return demod_bits