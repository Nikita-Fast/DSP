import copy
from typing import Any, List, Dict
import numpy as np
# TODO на мой взгляд сторонних библиотек быть не должно. Если есть острая необходимость ее надо аргументировать
import commpy.channelcoding as cc
from matplotlib import pyplot as plt
import uuid


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


class MethodCallDescription:
    def __init__(self, method_name: str, inputs: List[int], outputs: List[int]):
        self.method_name = method_name
        self.inputs = inputs
        self.outputs = outputs


# TODO вход - это список (пример демодулятора data = [[info], [snr]] ) выход - это список (пример блок АБГШ - [[data], [snr]])
# TODO класс должен быть абстрактным
class Block:

    def __init__(self):
        self.block_output = None
        self.id = uuid.uuid4()

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def process1(self, data: List, config: List[MethodCallDescription]) -> None:
        """Можно считать что параметр data это данные со всех входов этого блока, т.е. это список списков.
        Далее пользователь сам настраивает в какие методы подавать данные с каких входов и на какие выходы
        подавать результаты с помощью параметра config
        """
        max_output_num = 0
        for method_descr in config:
            max_output_num = max(max_output_num, *method_descr.outputs)

        block_output = [[] for i in range(max_output_num + 1)]

        for method_descr in config:
            data_copy = copy.deepcopy(data)
            # как в метод вызываемый по имени передать входные аргументы?
            # с помощью * перед списком аргументов

            data_for_method = [data_copy[i] for i in method_descr.inputs]
            method = getattr(self, method_descr.method_name)

            args = []
            for sub_list in data_for_method:
                args.append(sub_list.pop())

            method_output = method(*args)

            for output_num in method_descr.outputs:
                block_output[output_num].append(method_output)

        self.block_output = copy.deepcopy(block_output)


class Connection:
    def __init__(self, from_block: Block, from_output: int, to_block: Block, to_input: int):
        self.from_block = from_block
        self.from_output = from_output
        self.to_block = to_block
        self.to_input = to_input


def get_connections_to(to_block: Block, connections: List[Connection]):
    proper_connections = []
    for connection in connections:
        if connection.to_block == to_block:
            proper_connections.append(connection)
    return proper_connections


# TODO модель мне кажется должна быть в отдельном модуле
class Model:

    # TODO блоки могут соединятся в произвольном порядке. Я думаю надо перечень блоков и лист соединений (или еще как то)
    def __init__(self, blocks: List[Block], connections: List[Connection], starting_blocks: List[Block],
                 block_configs: Dict[uuid.UUID, List[MethodCallDescription]], final_block: Block):
        """Может ли стартовых блоков быть несколько?
        Например, несколько кодеров, стоящих параллельно, а не последовательно"""
        self.blocks = blocks
        self.connections = connections
        self.starting_blocks = starting_blocks
        self.block_configs = block_configs
        self.final_block = final_block

    # TODO поток данных должен идти в соответствии с листом соединений (что если у блока есть обратная связь )
    def process(self, data: np.ndarray) -> List:

        # возможно стоит создать специальный блок, соединенный со стартовыми, который выдаст начальные данные
        initial_data_block = Block()
        initial_data_block.block_output = [[data]]
        for starting_block in self.starting_blocks:
            connection = Connection(initial_data_block, 0, starting_block, 0)
            self.connections.append(connection)

        curr_blocks = self.starting_blocks

        # как прервать вечный цикл? По-идее тогда, когда финальный блок завершит обработку.
        # может ли финальных блоков быть несколько?
        while True:
            for block in curr_blocks:
                # нужно получить входные данные для блока от других предыдущих блоков или
                # получить начальные данные от моделирующей системы

                connections_to = get_connections_to(block, self.connections)

                block_input = [[] for i in connections_to]

                # нужно определить как будут браться данные из initial_data_block
                # например можно удалять соединение между initial_data_block и стартовым
                # блоком после того, как стартовые данные были извлечены
                for connection in connections_to:
                    if connection.from_block.block_output is not None:
                        block_input[connection.to_input] = connection.from_block.block_output[connection.from_output]
                        # удаляем соединение между initial_data_block и стартовым
                        # блоком после того, как стартовые данные были извлечены
                        # делаем так, чтобы брать начаьыне данные лишь один раз
                        # todo открыт вопрос о распределении стартовых данных между стартовыми блоками
                        if connection.from_block == initial_data_block:
                            self.connections.remove(connection)

                # входные данные для блока собраны, можно приступать к обработке
                block.process1(block_input, self.block_configs[block.id])

                # если блок финальный, то возвращаем результат его работы
                if block == self.final_block:
                    return block.block_output

            # обработали все текущие блоки. нужно сформировать новую группу.
            # выбираем те блоки в которые идут связи из текущих блоков
            next_blocks = []
            for connection in self.connections:
                if curr_blocks.__contains__(connection.from_block) and not next_blocks.__contains__(connection.to_block):
                    next_blocks.append(connection.to_block)

            curr_blocks = next_blocks

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
            # todo добавить специальный информационный блок, который будет передавать другим блокам ebn0_db, ... и т.д.
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
        return BERComputationResult(ber_points.copy(), 'default_name')

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
        super().__init__()
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation

    def process(self, data: np.ndarray) -> np.ndarray:
        pass


class BlockChannel(Block):

    def __init__(self, information_bits_per_symbol):
        super().__init__()
        self.information_bits_per_symbol = information_bits_per_symbol

    def process(self, data: np.ndarray, ebn0_db) -> np.ndarray:
        """Добавляем к сигналу необходимое кол-во шума"""
        pass

    def calc_noise_variance(self, ebn0_db):
        pass


class BlockDemodulator(Block):
    def __init__(self, bits_per_symbol, constellation: np.ndarray, mode='hard'):
        super().__init__()
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation
        self.mode = mode

    def process(self, data: np.ndarray, noise_variance=None) -> np.ndarray:
        """Демодулируем символы, возвращая биты в режиме 'hard' и llr-ы в режиме 'soft'.
        В мягком режиме требуется передать значение дисперсии шума"""
        pass


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
