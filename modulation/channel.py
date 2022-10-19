import numpy as np


def power(sig):
    """ Рассчитываем мощность сигнала."""
    return (((np.abs(sig)) ** 2).sum()) / len(sig)


class AWGNChannel:
    """Класс описывающий канал с АБГШ"""

    def __calc_noise_power(self, ebn0_db: int, symbols, modulation_bits_per_symbol: int, code_rate = 1):
        """ Рассчитываем необходимую мощность шума для переданного массива символов и требуемого EB_N0_db."""
        Es_N0_dB = ebn0_db + 10 * np.log10(modulation_bits_per_symbol * code_rate)
        SNR_dB = Es_N0_dB
        SNR = 10 ** (SNR_dB / 10)
        return power(symbols) / SNR

    def add_noise(self, symbols, ebn0_db: int, modulation_bits_per_symbol: int, code_rate = 1):
        """Добавляем к массиву символов необходимое количество шума, позволяющее
         получить требуемый уровень EB_N0_db"""
        p = self.__calc_noise_power(ebn0_db, symbols, modulation_bits_per_symbol, code_rate)

        symbols_num = len(symbols)
        awgn = np.sqrt(p / 2) * np.random.randn(symbols_num) + 1j * np.sqrt(p / 2) * np.random.randn(
            symbols_num)
        return symbols + awgn
