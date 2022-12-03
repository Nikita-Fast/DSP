from typing import Any

import commpy.channelcoding
import numpy as np
# from commpy.channelcoding import Trellis
import commpy.channelcoding as cc


class Coder:
    def __init__(self):
        pass

    def get_code_rate(self) -> float:
        pass

    def encode(self, input_bits) -> Any:
        pass


class ConvolutionalCoder(Coder):

    def __init__(self, memory=None, g_matrix=None, trellis: cc.Trellis = None):
        """Конструктор либо получает готовый треллис, либо создает его. Треллис полностью описывает кодер"""
        super().__init__()
        if trellis is None:
            if memory is not None and g_matrix is not None:
                self.trellis = cc.Trellis(memory, g_matrix)
            else:
                raise ValueError("Для описания кодера нужен либо треллис, либо память и генераторная матрица "
                                 "для построения треллиса")
        else:
            self.trellis = trellis

    def get_code_rate(self) -> float:
        return self.trellis.k / self.trellis.n

    def encode(self, input_bits) -> Any:
        return cc.conv_encode(input_bits, self.trellis)


class Modulator:
    def __init__(self, bits_per_symbol, constellation: np.ndarray):
        """

        :param bits_per_symbol: Количество битов в одном символе
        :param constellation: Список символов в созвездии, представленных комплексными числами. При переводе позици
        символа внутри списка в двоичную с.с. получим битовую последовательность, соответствующую данному символу
        """
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation
        pass

    def modulate(self, input_bits) -> Any:
        pass


class Channel:

    def add_noise(self, symbols, ebn0_db, information_bits_per_symbol) -> Any:
        pass

    def calc_noise_variance(self, ebn0_db, information_bits_per_symbol) -> float:
        pass


class Demodulator:
    def __init__(self, bits_per_symbol, constellation: np.ndarray, mode='hard'):
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation
        self.mode = mode

    def demodulate_hard(self, symbols) -> Any:
        pass

    def demodulate_soft(self, symbols, noise_variance) -> Any:
        pass


class Decoder:
    def decode(self, bits, llrs=None) -> Any:
        pass


class ConvolutionalDecoder(Decoder):
    def __init__(self, trellis: cc.Trellis):
        self.trellis = trellis

    @classmethod
    def from_coder(cls, coder: ConvolutionalCoder):
        return cls(coder.trellis)

    def decode(self, bits, llrs=None) -> Any:
        if llrs is None:
            return cc.viterbi_decode(bits, self.trellis)
        else:
            return cc.viterbi_decode(llrs, self.trellis, decoding_type='unquantized')
        pass