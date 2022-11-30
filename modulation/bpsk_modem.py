import numpy as np


class BPSKModem:

    def __init__(self):
        self.constellation = np.array([-1, 1])
        self.bits_per_symbol = 1
        self.order = 2
        self.name = "BPSK"

    def modulate(self, bits):
        return self.constellation[bits]

    def demodulate(self, symbols, demod_type='hard', noise_var=0.0, use_formula=False):
        if demod_type == 'hard':
            return np.abs(symbols[:, None] - self.constellation[None, :]).argmin(axis=1)
        elif demod_type == 'unquantized':
            return 2 * np.real(symbols) / noise_var
        else:
            raise ValueError("demod_type must be 'hard' or 'unquantized'")
