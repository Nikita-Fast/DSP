from typing import List, Optional, Tuple

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
        self.trellis = cc.Trellis(np.array([K - 1]), g_matrix, polynomial_format='MSB')

    def __str__(self):
        return str(self.g_matrix)


class SystemDescription:
    def __init__(self, coder: Optional[Encoder], qam_modulator: QAMModulator):
        self.coder = coder
        self.modulator = qam_modulator
        self.demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)


class ComputationParameters:
    def __init__(self, errors_threshold: int, max_processed_bits: int, enb0_range,
                 bits_process_per_iteration=10_000):
        self.errors_threshold = errors_threshold
        self.max_processed_bits = max_processed_bits
        self.ebn0_range = enb0_range
        self.bits_process_per_iteration = bits_process_per_iteration


class ComputationConfiguration:
    def __init__(self, sys_descr: SystemDescription, params: ComputationParameters):
        self.system_description = sys_descr
        self.params = params


class BERComputationResult:
    def __init__(self, ber_points: List[float], description: str):
        self.ber_points = ber_points
        self.description = description


class BERComputationResultConcurrent:
    def __init__(self, ber_points: List[Tuple[float, int]], description: str):
        self.ber_points = ber_points
        self.description = description


def compose_results_of_parallel_computations(results: List[BERComputationResultConcurrent]):
    ber_points = []

    for res in results:
        ber_points.extend(res.ber_points)
    ber_points.sort(key=lambda tup: tup[1])

    ber_points, _ = zip(*ber_points)

    return BERComputationResult(ber_points, results[0].description)


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
    for ebn0 in params.ebn0_range:
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
                out_bits = cc.viterbi_decode(demod_bits, trellis, decoding_type='hard')

            bits_processed += params.bits_process_per_iteration
            bit_errors += count_bit_errors(input_bits, out_bits[:len(input_bits)])

            print("ebn0 = %d, bits_processed = %d, errs = %d, appr_BER = %.7f"
                  % (ebn0, bits_processed, bit_errors, (bit_errors / bits_processed)))

        ber = bit_errors / bits_processed
        ber_points.append(ber)

        # np.save(name, ber_points)

    return BERComputationResult(ber_points.copy(), name)


def calc_ber_of_many_systems(systems: List[SystemDescription], params: List[ComputationParameters]):
    results = []

    for system, parameters in zip(systems, params):
        results.append(calc_ber_of_one_system(system, parameters))

    return results


def calc_ber_of_one_configuration(configuration: ComputationConfiguration):
    ber_points = []

    system = configuration.system_description
    params = configuration.params

    awgn_channel = AWGNChannel()
    coder = system.coder
    trellis = None
    modulator = system.modulator
    demodulator = system.demodulator

    order = 2 ** modulator.bits_per_symbol
    code = str(coder)
    name = "QAM-%d-%s-balanced-volume" % (order, code)

    print("Computing exact BER for " + name)
    for ebn0 in params.ebn0_range:
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
                out_bits = cc.viterbi_decode(demod_bits, trellis, decoding_type='hard')

            bits_processed += params.bits_process_per_iteration
            bit_errors += count_bit_errors(input_bits, out_bits[:len(input_bits)])

            print("ebn0 = %d, bits_processed = %d, errs = %d, appr_BER = %.7f"
                  % (ebn0, bits_processed, bit_errors, (bit_errors / bits_processed)))

        ber = bit_errors / bits_processed
        ber_points.append((ber, ebn0))

        # np.save(name, ber_points)

    return BERComputationResultConcurrent(ber_points.copy(), name)


coder1 = Encoder(K=3, g_matrix=np.array([[5, 7]]))
coder2 = Encoder(K=5, g_matrix=np.array([[35, 13]]))
coder3 = Encoder(K=6, g_matrix=np.array([[75, 53]]))
coder4 = Encoder(K=7, g_matrix=np.array([[133, 171]]))
coder5 = Encoder(K=8, g_matrix=np.array([[371, 247]]))

qam_modulator = QAMModulator(bits_per_symbol=2, bit_mapping=qam_modulation.gray_codes(2))
system0 = SystemDescription(None, qam_modulator)

system_75_53 = SystemDescription(coder3, qam_modulator)
system_171_133 = SystemDescription(coder4, qam_modulator)
system_371_247 = SystemDescription(coder5, qam_modulator)


from multiprocessing import Pool

p0 = ComputationParameters(2500, 7_000_00, [7], 250_000)
p1 = ComputationParameters(2500, 8_000_00, [8], 250_000)
p2 = ComputationParameters(2500, 9_000_00, [9], 250_000)
p3 = ComputationParameters(2500, 10_000_00, [10], 250_000)
p4 = ComputationParameters(2500, 5_000_00, [0,1,2,3,4,5,6], 250_000)

configs1 = [
    ComputationConfiguration(system0, p0),
    ComputationConfiguration(system0, p1),
    ComputationConfiguration(system0, p2),
    ComputationConfiguration(system0, p3),
    ComputationConfiguration(system0, p4),
]

p0 = ComputationParameters(50_000, 1_00_00, [0,1,2,3], 50_000)
p1 = ComputationParameters(2500, 2_50_00, [4], 10_000)
p2 = ComputationParameters(2500, 5_00_00, [5], 10_000)
p3 = ComputationParameters(2500, 7_50_00, [6], 10_000)
p4 = ComputationParameters(2500, 10_00_00, [7], 10_000)

configs2 = [
    ComputationConfiguration(system_171_133, p0),
    ComputationConfiguration(system_171_133, p1),
    ComputationConfiguration(system_171_133, p2),
    ComputationConfiguration(system_171_133, p3),
    ComputationConfiguration(system_171_133, p4),
]

# if __name__ == '__main__':
#     with Pool(5) as p:
#         t = time.time()
#         results = p.map(calc_ber_of_one_configuration, configs1)
#         print(time.time() - t)
#         res1 = compose_results_of_parallel_computations(results)
#
#         # results = p.map(calc_ber_of_one_configuration, configs2)
#         # res2 = compose_results_of_parallel_computations(results)
#
#         plot_ber_computation_results([res1])


def calc_ber(coder, modulator, demodulator, demod_type, params: ComputationParameters):
    ber_points = []

    awgn_channel = AWGNChannel()
    trellis = None

    order = 2 ** modulator.bits_per_symbol
    code = str(coder)
    mod_name = modulator.get_name()
    name = "%s-%d-%s-%s-balanced-volume" % (mod_name, order, demod_type, code)

    print("Computing exact BER for " + name)
    for ebn0 in params.ebn0_range:
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

            noise_var = channel.calc_noise_variance(ebn0, modulator.bits_per_symbol, code_rate)

            demodulated = demodulator.demodulate(dirty_sig, demod_type, noise_var)
            out_bits = demodulated

            if coder is not None:
                out_bits = cc.viterbi_decode(demodulated, trellis, decoding_type=demod_type)

            bits_processed += params.bits_process_per_iteration
            bit_errors += count_bit_errors(input_bits, out_bits[:len(input_bits)])

            print("ebn0 = %d, bits_processed = %d, errs = %d, appr_BER = %.7f"
                  % (ebn0, bits_processed, bit_errors, (bit_errors / bits_processed)))

        ber = bit_errors / bits_processed
        ber_points.append(ber)

        np.save(name, ber_points)

    return BERComputationResult(ber_points.copy(), name)


coder_7_5 = Encoder(K=3, g_matrix=np.array([[5, 7]]))
coder_75_53 = Encoder(K=6, g_matrix=np.array([[75, 53]]))
bpsk_modem = bpsk_modem.BPSKModem()
p1 = ComputationParameters(2500, 50_000_000, [0,1,2,3,4,5,6,7,8,9,10], 100_000)
p2 = ComputationParameters(2500, 6_000_000, [0,1,2,3,4,5,6,7,8], 60_000)
p3 = ComputationParameters(2500, 6_000_000, [0,1,2,3,4,5,6,7], 60_000)

r1 = calc_ber(None, bpsk_modem, bpsk_modem, 'hard', p1)
r2 = calc_ber(coder_75_53, bpsk_modem, bpsk_modem, 'hard', p2)
r3 = calc_ber(coder_75_53, bpsk_modem, bpsk_modem, 'unquantized', p3)

plot_ber_computation_results([r1, r2, r3])
