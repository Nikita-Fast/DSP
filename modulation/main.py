from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import time

import qam_modulation
from channel import AWGNChannel
from qam_demodulation import QAMDemodulator
from qam_modulation import QAMModulator
import commpy.channelcoding.convcode as cc
from modulation_type import ModulationType


def gen_bits(seed, bits_num: int):
    np.random.seed(0)
    return np.random.randint(low=0, high=2, size=bits_num)


def count_bit_errors(input_bits, output_bits):
    return np.sum(np.abs(input_bits - output_bits), dtype=int)


def ber_calc(a, b):
    bit_errors = np.sum(np.abs(a - b), dtype=int)
    ber = np.mean(np.abs(a - b))
    return bit_errors, ber


class Encoder:
    def __init__(self, K, g_matrix):
        self.K = K
        self.g_matrix = g_matrix
        self.trellis = cc.Trellis(np.array([K - 1]), g_matrix)

    def __str__(self):
        return str(self.g_matrix)


class SystemDescription:
    def __init__(self, coder: Optional[Encoder], qam_modulator: QAMModulator):
        self.coder = coder
        self.modulator = qam_modulator
        self.demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)


class ComputationParameters:
    def __init__(self, errors_threshold: int, max_processed_bits: int, max_ebn0: int, bits_process_per_iteration: int):
        self.errors_threshold = errors_threshold
        self.max_processed_bits = max_processed_bits
        self.max_ebn0 = max_ebn0
        self.bits_process_per_iteration = bits_process_per_iteration


class BERComputationResult:
    def __init__(self, ber_points, description: str):
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
    coder = system.coder
    trellis = None
    modulator = system.modulator
    demodulator = system.demodulator

    order = 2 ** modulator.bits_per_symbol
    code = str(coder)
    name = "QAM-%d-%s-balanced-volume" % (order, code)

    print("Computing exact BER for " + name)
    for ebn0 in range(params.max_ebn0 + 1):
        bit_errors = 0
        bits_processed = 0

        while bit_errors < params.errors_threshold and bits_processed < params.max_processed_bits:
            input_bits = np.random.randint(low=0, high=2, size=params.bits_process_per_iteration)
            bits = input_bits

            code_rate = 1.0
            if coder is not None:
                trellis = coder.trellis
                bits = cc.conv_encode(input_bits, trellis)
                code_rate = trellis.k / trellis.n

            mod_signal = modulator.modulate(bits)
            dirty_sig = awgn_channel.add_noise(mod_signal, ebn0, modulator.bits_per_symbol, code_rate)
            demod_bits = demodulator.demodulate(dirty_sig)
            out_bits = demod_bits

            if coder is not None:
                tb_depth = 5 * coder.K
                out_bits = cc.viterbi_decode(demod_bits, trellis, tb_depth, decoding_type='hard')

            bits_processed += params.bits_process_per_iteration
            bit_errors += count_bit_errors(input_bits, out_bits[:len(input_bits)])

            print("ebn0 = %d, bits_processed = %d, errs = %d, appr_BER = %.7f"
                  % (ebn0, bits_processed, bit_errors, (bit_errors / bits_processed)))

        ber = bit_errors / bits_processed
        ber_points.append(ber)

        np.save(name, ber_points)

    return BERComputationResult(ber_points.copy(), name)


def calc_ber_of_many_systems(systems: List[SystemDescription], params: List[ComputationParameters]):
    results = []

    for system, parameters in zip(systems, params):
        results.append(calc_ber_of_one_system(system, parameters))

    return results


coder1 = Encoder(K=3, g_matrix=np.array([[5, 7]]))
coder2 = Encoder(K=5, g_matrix=np.array([[35, 13]]))
coder3 = Encoder(K=7, g_matrix=np.array([[171, 133]]))

qam_modulator = QAMModulator(bits_per_symbol=2, bit_mapping=qam_modulation.gray_codes(2))
system1 = SystemDescription(None, qam_modulator)
system2 = SystemDescription(coder1, qam_modulator)
system3 = SystemDescription(coder2, qam_modulator)
system4 = SystemDescription(coder3, qam_modulator)

params_no_coder = ComputationParameters(errors_threshold=2500, max_processed_bits=25_000_000,
                                max_ebn0=10, bits_process_per_iteration=50_000)

params_coder = ComputationParameters(errors_threshold=2000, max_processed_bits=2_500_000,
                                max_ebn0=10, bits_process_per_iteration=10_000)

results = calc_ber_of_many_systems([system1,
                                    system2,
                                    system3,
                                    system4],
                                   [params_no_coder,
                                    params_coder,
                                    params_coder,
                                    params_coder])
plot_ber_computation_results(results)

# curve = np.load('QAM-8-None-balanced-volume.npy')
# plot_ber_curves([curve], ["test"])
