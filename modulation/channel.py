import numpy as np


def power(sig):
    """ Рассчитываем мощность сигнала."""
    return (((np.abs(sig)) ** 2).sum()) / len(sig)


class AWGNChannel:
    """Класс описывающий канал с АБГШ"""

    def __calc_noise_power(self, EB_N0_db, symbols, bits_per_symbol):
        """ Рассчитываем необходимую мощность шума для переданного массива символов и требуемого EB_N0_db."""
        Es_N0_dB = EB_N0_db + 10 * np.log10(bits_per_symbol)
        SNR_dB = Es_N0_dB
        SNR = 10 ** (SNR_dB / 10)
        return power(symbols) / SNR

    def add_noise(self, symbols, EB_N0_db, bits_per_symbol):
        """Добавляем к массиву символов необходимое количество шума, позволяющее
         получить требуемый уровень EB_N0_db"""
        p = self.__calc_noise_power(EB_N0_db, symbols, bits_per_symbol)

        symbols_num = len(symbols)
        awgn = np.sqrt(p / 2) * np.random.randn(symbols_num) + 1j * np.sqrt(p / 2) * np.random.randn(
            symbols_num)
        return symbols + awgn
