import numpy as np

from interface import Modulator, Demodulator


class BPSKModem(Modulator, Demodulator):

    def __init__(self, mode='hard'):
        super().__init__(1, np.array([-1, 1]))
        self.order = 2
        self.name = "BPSK"
        self.mode = mode

    def modulate(self, bits):
        return self.constellation[bits]

    def demodulate(self, symbols, demod_type='hard', noise_var=0.0, use_formula=False):
        if demod_type == 'hard':
            return np.abs(symbols[:, None] - self.constellation[None, :]).argmin(axis=1)
        elif demod_type == 'unquantized':
            return 2 * np.real(symbols) / noise_var
        else:
            raise ValueError("demod_type must be 'hard' or 'unquantized'")

    def demodulate_hard(self, symbols):
        return np.abs(symbols[:, None] - self.constellation[None, :]).argmin(axis=1)

    def demodulate_soft(self, symbols, noise_variance):
        return 2 * np.real(symbols) / noise_variance
