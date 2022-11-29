from typing import List

import commpy
import numpy as np
import matplotlib.pyplot as plt
import time

import bpsk_modem
import channel
import qam_modulation
from channel import AWGNChannel
from qam_demodulation import QAMDemodulator
from qam_modulation import QAMModulator
import commpy.channelcoding.convcode as cc


def count_bit_errors(input_bits, output_bits):
    return np.sum(np.abs(input_bits - output_bits), dtype=int)


def ber_calc(a, b):
    bit_errors = np.sum(np.abs(a - b), dtype=int)
    ber = np.mean(np.abs(a - b))
    return bit_errors, ber


class Code:
    def __init__(self, K, g_matrix):
        self.K = K
        self.g_matrix = g_matrix
        self.trellis = cc.Trellis(np.array([K - 1]), g_matrix, polynomial_format='MSB')
        self.code_rate = self.trellis.k / self.trellis.n

    def __str__(self):
        return str(self.g_matrix)


class SystemDescription:
    def __init__(self, modulator, demodulater, demod_type='hard', code: Code = None, use_formula=False):
        self.code = code
        self.modulator = modulator
        self.demodulator = demodulater
        self.demod_type = demod_type
        self.use_formula = use_formula


class ComputationParameters:
    def __init__(self, errors_threshold: int, max_processed_bits: int, enb0_range,
                 bits_process_per_iteration=10_000):
        self.errors_threshold = errors_threshold
        self.max_processed_bits = max_processed_bits
        self.ebn0_range = enb0_range
        self.bits_process_per_iteration = bits_process_per_iteration


class ComputationConfiguration:
    def __init__(self, system_description: SystemDescription, params: ComputationParameters):
        self.system_description = system_description
        self.params = params


class BERComputationResult:
    def __init__(self, ber_points: List[float], description: str):
        self.ber_points = ber_points
        self.description = description


def plot_ber_computation_results(results: List[BERComputationResult]):
    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")

    for res in results:
        plt.plot(res.ber_points, '--o', label=res.description)
        plt.legend()
    plt.show()


def calc_ber_of_one_system(system: SystemDescription, params: ComputationParameters):
    ber_points = []

    awgn_channel = AWGNChannel()

    code = system.code
    modulator = system.modulator
    demodulator = system.demodulator
    trellis = None

    code_name = 'no-code' if code is None else str(code)
    name = "%s-%s-%s-use_formula=%s" % (modulator.name, code_name, system.demod_type, str(system.use_formula))

    print("Computing exact BER for %s" % name)
    for ebn0 in params.ebn0_range:
        bit_errors = 0
        bits_processed = 0

        while bit_errors < params.errors_threshold and bits_processed < params.max_processed_bits:
            input_bits = np.random.randint(low=0, high=2, size=params.bits_process_per_iteration)
            bits = input_bits

            code_rate = 1.0
            if code is not None:
                trellis = code.trellis
                bits = cc.conv_encode(input_bits, trellis)
                code_rate = code.code_rate

            mod_signal = modulator.modulate(bits)
            dirty_sig = awgn_channel.add_noise(mod_signal, ebn0, modulator.bits_per_symbol, code_rate)

            noise_var = channel.calc_noise_variance(ebn0, modulator.bits_per_symbol, code_rate)
            demod_bits = demodulator.demodulate(dirty_sig, system.demod_type, noise_var, use_formula=system.use_formula)
            out_bits = demod_bits

            if code is not None:
                out_bits = cc.viterbi_decode(demod_bits, trellis, decoding_type=system.demod_type)

            bits_processed += params.bits_process_per_iteration
            bit_errors += count_bit_errors(input_bits, out_bits[:len(input_bits)])

            print("ebn0 = %d, bits_processed = %d, errs = %d, appr_BER = %.7f"
                  % (ebn0, bits_processed, bit_errors, (bit_errors / bits_processed)))

        ber = bit_errors / bits_processed
        ber_points.append(ber)

    return BERComputationResult(ber_points.copy(), name)


def calc_ber_of_many_systems(systems: List[SystemDescription], params: List[ComputationParameters]):
    results = []

    for system, parameters in zip(systems, params):
        results.append(calc_ber_of_one_system(system, parameters))

    return results


# Код моделирования
# coder_75_53 = Code(K=6, g_matrix=np.array([[75, 53]]))
coder_7_5 = Code(K=3, g_matrix=np.array([[7, 5]]))
coder_15_11 = Code(K=4, g_matrix=np.array([[15, 11]]))
# bpsk_modem = bpsk_modem.BPSKModem()
qam16_modem = QAMModulator(bits_per_symbol=4)

p1 = ComputationParameters(2500, 250_000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 25_000)
# p2 = ComputationParameters(2500, 6_000_000, [0, 1, 2, 3, 4, 5, 6, 7, 8], 60_000)
# p3 = ComputationParameters(2500, 6_000_000, [0, 1, 2, 3, 4, 5, 6, 7], 60_000)

# sys1 = SystemDescription(bpsk_modem, bpsk_modem, 'hard', code=None)
# sys2 = SystemDescription(bpsk_modem, bpsk_modem, 'hard', code=coder_75_53)
# sys3 = SystemDescription(bpsk_modem, bpsk_modem, 'unquantized', code=coder_75_53)

# results = calc_ber_of_many_systems(systems=[sys1, sys2, sys3], params=[p1, p2, p3])

sys1 = SystemDescription(qam16_modem, QAMDemodulator.from_qam_modulator(qam16_modem), 'hard', code=None)

sys2 = SystemDescription(qam16_modem, QAMDemodulator.from_qam_modulator(qam16_modem), 'hard', code=coder_15_11)

sys3 = SystemDescription(qam16_modem, QAMDemodulator.from_qam_modulator(qam16_modem),
                         'unquantized', code=coder_15_11, use_formula=True)

sys4 = SystemDescription(qam16_modem, QAMDemodulator.from_qam_modulator(qam16_modem), 'unquantized', code=coder_15_11)

# results = calc_ber_of_many_systems(systems=[sys1, sys2, sys3, sys4], params=[p1, p1, p1, p1])
#
# plot_ber_computation_results(results)


# class TCM:
#     def __init__(self):
#         from numpy import sin
#         from numpy import cos
#         from numpy import pi
#         self.constellation = np.array([1 + 1j * 0,
#                                        cos(3 * pi / 4) + 1j * sin(3 * pi / 4),
#                                        cos(1 * pi / 4) + 1j * sin(1 * pi / 4),
#                                        cos(2 * pi / 4) + 1j * sin(2 * pi / 4),
#                                        cos(4 * pi / 4) + 1j * sin(4 * pi / 4),
#                                        cos(7 * pi / 4) + 1j * sin(7 * pi / 4),
#                                        cos(5 * pi / 4) + 1j * sin(5 * pi / 4),
#                                        cos(6 * pi / 4) + 1j * sin(6 * pi / 4)])
#         # self.coder = Code(K=3, g_matrix=np.array([[7, 5]]))
#
#         # self.transition_table = np.array([[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]])
#         # self.output_table = np.array([[0, 3, 4, 7], [2, 1, 6, 5], [3, 0, 7, 4], [1, 2, 5, 6]])
#         self.trellis = cc.Trellis(memory=np.array([2]), g_matrix=np.array([[7,5]]))
#         self.trellis.next_state_table = np.array([[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]])
#         self.trellis.output_table = np.array([[0, 3, 4, 7], [2, 1, 6, 5], [3, 0, 7, 4], [1, 2, 5, 6]])
#         # self.trellis.output_table = self.constellation[self.output_table]
#         self.trellis.number_inputs = 4
#         self.trellis.k = 2
#         self.trellis.n = 3
#         self.output_symbols_table = self.constellation[self.trellis.output_table]
#
#
#     def encode(self, bits):
#         coded_bits = cc.conv_encode(bits, self.trellis)
#         values = np.empty(len(coded_bits)//3, dtype=int)
#         for i in range(len(coded_bits)//3):
#             c1 = coded_bits[3 * i]
#             c2 = coded_bits[3*i+1]
#             c3 = coded_bits[3 * i + 2]
#             value = (c1 << 2) + (c2 << 1) + c3
#             values[i] = value
#         print("tcm-coder-output: ", values)
#         return self.constellation[values]
#
#
#     def decode(self, r):
#         tb_depth = 3 * 5
#
#         states_num = 4
#         path_metrics = np.full((states_num, 2), np.inf)
#         path_metrics[0][0] = 0
#         paths = np.empty((states_num, tb_depth), 'int')
#         paths[0][0] = 0
#
#         pred_states = [0]
#
#         metrics_table = np.full((16, tb_depth), np.inf)
#         time = 0
#
#         next_state_table = self.trellis.next_state_table
#         for i in range(len(r)):
#             for pred_state in pred_states:
#                 for k in range(4):
#                     symbol = self.output_symbols_table[pred_state][k]
#                     dist = np.abs(r[i] - symbol)
#                     metrics_table[4*pred_state+k][time] = dist
#             # отбросим большую метрику в каждом параллельном переходе
#             for pred_state in pred_states:
#                 dst_states = np.unique(next_state_table[pred_state])
#                 for state in dst_states:
#                     idx = np.where(next_state_table[pred_state] == state)[0]
#                     i0 = idx[0]
#                     i1 = idx[1]
#                     offset = 4*pred_state
#                     if metrics_table[offset+i0][time] <= metrics_table[offset+i1][time]:
#                         metrics_table[offset+i1][time] = np.inf
#                     else:
#                         metrics_table[offset+i0][time] = np.inf
#             # считаем метрику каждого состояния
#             for state in range(4):
#                 incoming_idx = np.argwhere(next_state_table == state)
#                 incoming_metrics = []
#                 for coords in incoming_idx:
#                     index = coords[0] * 4 + coords[1]
#                     if (metrics_table[index][time] < np.inf):
#                         incoming_metrics.append(metrics_table[index][time])
#                 if len(incoming_metrics) == 1:
#                     state_metrics[state] =

from tcm import TCM


tcm = TCM()


bits = np.array([0, 1, 1, 0, 1, 0])
symbols = tcm.encode(bits)
print(symbols)
tcm.decode(np.array([-5,1,-7,3]))
metrics = tcm.extract_metrics_from_transition_table()
print(metrics)
print(tcm.state_metrics)
print("end")
# plt.plot(symbols.real, symbols.imag, 'o')
# plt.show()