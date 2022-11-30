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

from tcm import TCM


def compare_arrays(arr1, arr2):
    errs = 0
    for i in range(min(len(arr1), len(arr2))):
        if arr1[i] != arr2[i]:
            errs = errs + 1
    return errs


tcm = TCM()

information_bits_to_transmit = 10_000
padding = 50

bits = np.zeros(information_bits_to_transmit + padding, int)
bits[0:10_000] = np.random.randint(low=0, high=2, size=information_bits_to_transmit)

symbols = tcm.encode(bits)
decoded_bits = tcm.decode(symbols)

print("bit errs: %i" % compare_arrays(bits[:information_bits_to_transmit], decoded_bits))
