import numpy as np


class BPSKModem:
    constellation = np.array([-1, 1])
    bits_per_symbol = 1

    def modulate(self, bits):
        return self.constellation[bits]

    def demodulate(self, symbols, demod_type='hard', noise_var=0.0):
        if demod_type == 'hard':
            return np.abs(symbols[:, None] - self.constellation[None, :]).argmin(axis=1)
        elif demod_type == 'unquantized':
            return 2 * np.real(symbols) / noise_var
        else:
            raise ValueError("demod_type must be 'hard' or 'unquantized'")

    def get_name(self):
        return "BPSK"
