from typing import Any, List
import numpy as np
import commpy.channelcoding as cc # TODO на мой взгляд сторонних библиотек быть не должно. Если есть острая необходимость ее надо аргументировать
from matplotlib import pyplot as plt


# TODO это явно не к этому файлу
class BERComputationResult:
    def __init__(self, ber_points: List[float], description: str):
        self.ber_points = ber_points
        self.description = description

    def plot(self):
        plt.yscale("log")
        plt.grid(visible='true')
        plt.xlabel("Eb/N0, dB")
        plt.ylabel("BER")

        plt.plot(self.ber_points, '--o', label=self.description)
        plt.legend()
        plt.show()

# TODO это явно не к этому файлу
class ComputationParameters:
    def __init__(self, errors_threshold: int, max_processed_bits: int, enb0_range,
                 bits_process_per_iteration=10_000):
        self.errors_threshold = errors_threshold
        self.max_processed_bits = max_processed_bits
        self.ebn0_range = enb0_range
        self.bits_process_per_iteration = bits_process_per_iteration


# TODO вход - это список (пример демодулятора data = [[info], [snr]] ) выход - это список (пример блок АБГШ - [[data], [snr]])
# TODO класс должен быть абстрактным
class Block:
    def process(self, data: np.ndarray) -> np.ndarray:
        pass


# TODO модель мне кажется должна быть в отдельном модуле
class Model:

    # TODO блоки могут соединятся в произвольном порядке. Я думаю надо перечень блоков и лист соединений (или еще как то )
    def __init__(self, blocks: List[Block], name: str = 'default_name'):
        self.blocks = blocks
        self.name = name

    # TODO поток данных должен идти в соответствии с листом соединений (что если у блок  есть обратная связь )
    def __process(self, data: np.ndarray) -> np.ndarray:
        """Пропускаем входные данные последовательно через все блоки и возвращаем результат"""
        output = np.copy(data)
        for block in self.blocks:
            output = block.process(output)
        return output

    # TODO это отдельно обсудим
    def do_modelling(self, params: ComputationParameters) -> BERComputationResult:
        """Осуществляем моделирование системы с учетом переданных параметров"""
        ber_points = []

        # как передать в блок, представляющий канал, требуемое значение ebn0?
        # Думаю это полный отстой, как сделать лучше?
        #############################################################
        channel_block = None
        demodulator_block = None

        for block in self.blocks:
            if isinstance(block, BlockChannel):
                channel_block = block
                break

        for block in self.blocks:
            if isinstance(block, BlockDemodulator):
                demodulator_block = block
                break
        ###############################################################
        for ebn0 in params.ebn0_range:
            bit_errors = 0
            bits_processed = 0

            # Думаю так писать плохо
            channel_block.set_ebn0_db(ebn0)
            demodulator_block.set_noise_variance(channel_block.calc_noise_variance())

            while bit_errors < params.errors_threshold and bits_processed < params.max_processed_bits:
                # могут ли входные данные быть не битами?
                input_data = self.__gen_bits(params.bits_process_per_iteration)
                output_data = self.__process(input_data)
                bit_errors += self.__count_errors(input_data, output_data)
                bits_processed += params.bits_process_per_iteration

                print("ebn0 = %d, bits_processed = %d, errs = %d, appr_BER = %.7f"
                      % (ebn0, bits_processed, bit_errors, (bit_errors / bits_processed)))

            ber = bit_errors / bits_processed
            ber_points.append(ber)
        return BERComputationResult(ber_points.copy(), self.name)

    def __gen_bits(self, size):
        return np.random.randint(low=0, high=2, size=size)

    def __count_errors(self, arr1, arr2):
        """Если длины сравниваемых массивов неравны, то сравниваются куски длинной равной длине меньшего из массивов"""
        errs = 0
        for i in range(min(len(arr1), len(arr2))):
            if arr1[i] != arr2[i]:
                errs = errs + 1
        return errs

# TODO все что ниже пока не надо. Это требует обсуждения . Мне кажется сначала надо разобраться с Block и  Model
class BlockModulator(Block):

    def __init__(self, bits_per_symbol, constellation: np.ndarray):
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation
        pass

    def process(self, data: np.ndarray) -> np.ndarray:
        pass


class BlockChannel(Block):

    def __init__(self, ebn0_db, information_bits_per_symbol):
        self.ebn0_db = ebn0_db
        self.information_bits_per_symbol = information_bits_per_symbol

    def process(self, data: np.ndarray) -> np.ndarray:
        """Добавляем к сигналу необходимое кол-во шума"""
        pass

    def set_ebn0_db(self, ebn0_db):
        self.ebn0_db = ebn0_db

    def calc_noise_variance(self):
        pass


class BlockDemodulator(Block):
    def __init__(self, bits_per_symbol, constellation: np.ndarray, mode='hard'):
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation
        self.mode = mode
        self.noise_variance = 1.0

    # Когда демодулятор работает в мягком режиме, ему необходимо знать дисперсию шума. Дисперсию шума может
    # вычислить метод класса AWGNChannel. Возможно поскольку сейчас при декодировании используется
    # алгоритм витерби, то можно обойтись без деления llr-ов на дисперсию шума. Но хочется решить эту проблему
    # заранее. Как в этот блок передавать значение ebn0 и метод по расчету дисперсии шума?
    def process(self, data: np.ndarray) -> np.ndarray:
        """Демодулируем символы, возвращая биты в режиме 'hard' и llr-ы в режиме 'soft'"""
        pass

    def set_noise_variance(self, noise_var):
        self.noise_variance = noise_var


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