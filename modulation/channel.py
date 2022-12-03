import numpy as np

from interface import Channel


def power(sig):
    """ Рассчитываем мощность сигнала."""
    return (((np.abs(sig)) ** 2).sum()) / len(sig)


def calc_noise_power(ebn0_db, symbols, information_bits_per_symbol):
    """ Рассчитываем необходимую мощность шума для переданного массива символов и требуемого EB_N0_db."""
    Es_N0_dB = ebn0_db + 10 * np.log10(information_bits_per_symbol)
    SNR_dB = Es_N0_dB
    SNR = 10 ** (SNR_dB / 10)
    return power(symbols) / SNR


class AWGNChannel(Channel):
    """Класс описывающий канал с АБГШ"""

    def add_noise(self, symbols, ebn0_db, information_bits_per_symbol):
        """Добавляем к массиву символов необходимое количество шума, позволяющее
         получить требуемый уровень EB/N0 выраженный в db"""
        p = calc_noise_power(ebn0_db, symbols, information_bits_per_symbol)

        symbols_num = len(symbols)
        awgn = np.sqrt(p / 2) * np.random.randn(symbols_num) + 1j * np.sqrt(p / 2) * np.random.randn(
            symbols_num)
        return symbols + awgn

    def calc_noise_variance(self, ebn0_db, information_bits_per_symbol) -> float:
        snr_db = ebn0_db + 10 * np.log10(information_bits_per_symbol)  # Signal-to-Noise ratio (in dB)
        noise_var = 10 ** (-snr_db / 10)  # noise variance (power)
        return noise_var
