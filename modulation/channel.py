import numpy as np


def power(sig):
    return (((np.abs(sig)) ** 2).sum()) / len(sig)


class AWGNChannel:
    # def __init__(self, EB_N0_db, bits_per_symbol):

    def __calc_noise_power(self, EB_N0_db, symbols, bits_per_symbol):
        Es_N0_dB = EB_N0_db + 10 * np.log10(bits_per_symbol)
        SNR_dB = Es_N0_dB
        SNR = 10 ** (SNR_dB / 10)
        return power(symbols) / SNR

    def add_noise(self, symbols, EB_N0_db, bits_per_symbol):
        p = self.__calc_noise_power(EB_N0_db, symbols, bits_per_symbol)

        symbols_num = len(symbols)
        awgn = np.sqrt(p / 2) * np.random.randn(symbols_num) + 1j * np.sqrt(p / 2) * np.random.randn(
            symbols_num)
        return symbols + awgn
