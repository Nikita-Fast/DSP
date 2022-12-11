# TODO модель мне кажется должна быть в отдельном модуле
from typing import List, Dict
from interface import Block, Connection, InformationBlock, MethodCallDescription
from utils import *


class Model:

    # TODO блоки могут соединятся в произвольном порядке. Я думаю надо перечень блоков и лист соединений (или еще как то)
    def __init__(self, blocks: List[Block],
                 connections: List[Connection],
                 starting_blocks: List[Block],
                 block_configs: Dict[Block, List[MethodCallDescription]],
                 final_block: Block,
                 info_block: InformationBlock):
        """Может ли стартовых блоков быть несколько?
        Например, несколько кодеров, стоящих параллельно, а не последовательно"""
        self.blocks = blocks
        self.connections = connections
        self.starting_blocks = starting_blocks
        self.block_configs = block_configs
        self.final_block = final_block

        self.info_block = info_block

    # TODO поток данных должен идти в соответствии с листом соединений (что если у блока есть обратная связь )
    def process(self, data: np.ndarray) -> List:

        # информационный блок кладет параметры моделирования на свои выходы
        self.info_block.execute(data=[], config=self.block_configs[self.info_block])

        # создаем специальный блок, соединенный со стартовыми, который выдаст начальные данные
        initial_data_block = Block()
        initial_data_block.block_output = [[data]]
        for starting_block in self.starting_blocks:
            connection = Connection(initial_data_block, 0, starting_block, 0)
            self.connections.append(connection)

        curr_blocks = self.starting_blocks

        # когда прервать вечный цикл? По-идее тогда, когда финальный блок завершит обработку.
        # может ли финальных блоков быть несколько?
        while True:
            for block in curr_blocks:
                # нужно получить входные данные для блока от других предыдущих блоков или
                # получить начальные данные от моделирующей системы
                connections_to = get_connections_to(block, self.connections)

                block_input = [[] for i in connections_to]

                # Нужно определить как будут браться данные из initial_data_block.
                # Сейчас соединение между initial_data_block и стартовым блоком удаляется после того,
                # как стартовые данные были извлечены.
                for connection in connections_to:
                    if connection.from_block.block_output is not None:
                        block_input[connection.to_input] = connection.from_block.block_output[connection.from_output]
                        # удаляем соединение между initial_data_block и стартовым блоком
                        # todo открыт вопрос о распределении стартовых данных между стартовыми блоками
                        if connection.from_block == initial_data_block:
                            self.connections.remove(connection)

                # входные данные для блока собраны, можно приступать к обработке
                block.execute(block_input, self.block_configs[block])

                # если блок финальный, то возвращаем все его выходы т.е. список списков. В каких-то выходах находится
                # результат работы блока
                if block == self.final_block:
                    return block.block_output

            # обработали все текущие блоки. нужно сформировать новую группу.
            # выбираем те блоки в которые идут связи из текущих блоков
            next_blocks = []
            for connection in self.connections:
                if curr_blocks.__contains__(connection.from_block) and not next_blocks.__contains__(
                        connection.to_block):
                    next_blocks.append(connection.to_block)

            curr_blocks = next_blocks

    # TODO это отдельно обсудим
    def do_modelling(self, params: ComputationParameters) -> BERComputationResult:
        ber_points = []
        for ebn0 in params.ebn0_range:
            bit_errors = 0
            bits_processed = 0

            # todo добавить специальный информационный блок, который будет передавать другим блокам ebn0_db, ... и т.д.
            self.info_block.save_info(ebn0_db=ebn0)

            while bit_errors < params.errors_threshold and bits_processed < params.max_processed_bits:
                # могут ли входные данные быть не битами?
                input_data = gen_bits(params.bits_process_per_iteration)

                output = self.process(input_data)

                # todo? добавить спец блок собирающий обработанные данные
                bit_errors += count_errors(input_data, output[0][0])
                bits_processed += params.bits_process_per_iteration

                print("ebn0 = %d, bits_processed = %d, errs = %d, appr_BER = %.7f"
                      % (ebn0, bits_processed, bit_errors, (bit_errors / bits_processed)))

            ber = bit_errors / bits_processed
            ber_points.append(ber)
        return BERComputationResult(ber_points.copy(), 'default_name')
