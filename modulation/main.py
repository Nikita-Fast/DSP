from typing import List
import matplotlib.pyplot as plt

# import bpsk_modem
import default_qam_constellations
from channel import AWGNChannel
from conv_coder import ConvCoder, ConvDecoder
from qam_demodulation import QAMDemodulator
from qam_modulation import QAMModulator
# from tcm import TCM
from interface import *
import commpy.channelcoding as cc


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


class ComputationParameters:
    def __init__(self, errors_threshold: int, max_processed_bits: int, enb0_range,
                 bits_process_per_iteration=10_000):
        self.errors_threshold = errors_threshold
        self.max_processed_bits = max_processed_bits
        self.ebn0_range = enb0_range
        self.bits_process_per_iteration = bits_process_per_iteration


def count_bit_errors(arr1, arr2):
    # return np.sum(np.abs(input_bits - output_bits), dtype=int)
    errs = 0
    for i in range(min(len(arr1), len(arr2))):
        if arr1[i] != arr2[i]:
            errs = errs + 1
    return errs


#
#
# def ber_calc(a, b):
#     bit_errors = np.sum(np.abs(a - b), dtype=int)
#     ber = np.mean(np.abs(a - b))
#     return bit_errors, ber
#
#
# class ComputationParameters:
#     def __init__(self, errors_threshold: int, max_processed_bits: int, enb0_range,
#                  bits_process_per_iteration=10_000):
#         self.errors_threshold = errors_threshold
#         self.max_processed_bits = max_processed_bits
#         self.ebn0_range = enb0_range
#         self.bits_process_per_iteration = bits_process_per_iteration
#
#
# class BERComputationResult:
#     def __init__(self, ber_points: List[float], description: str):
#         self.ber_points = ber_points
#         self.description = description
#
#
# def plot_ber_computation_results(results: List[BERComputationResult]):
#     plt.yscale("log")
#     plt.grid(visible='true')
#     plt.xlabel("Eb/N0, dB")
#     plt.ylabel("BER")
#
#     for res in results:
#         plt.plot(res.ber_points, '--o', label=res.description)
#         plt.legend()
#     plt.show()
#
#
# def simulate(coder: Coder = None, modulator: Modulator = None, channel: Channel = None, demodulator: Demodulator = None,
#              decoder: Decoder = None,
#              params: ComputationParameters = None, name="unnamed"):
#     if params is None:
#         params = ComputationParameters(2500, 400_000, [0, 1, 2, 3, 4, 5, 6, 7, 8], 50_000)
#
#     ber_points = []
#     code_rate = 1.0
#     bits_per_symbol = 1
#     noise_variance = 1.0
#
#     print("Computing exact BER for %s" % name)
#     for ebn0 in params.ebn0_range:
#         bit_errors = 0
#         bits_processed = 0
#
#         while bit_errors < params.errors_threshold and bits_processed < params.max_processed_bits:
#             message_bits = np.random.randint(low=0, high=2, size=params.bits_process_per_iteration)
#             data = np.copy(message_bits)
#
#             if coder is not None:
#                 code_rate = coder.get_code_rate()
#                 data = coder.encode(data)
#
#             if modulator is not None:
#                 bits_per_symbol = modulator.bits_per_symbol
#                 data = modulator.modulate(data)
#
#             if channel is not None:
#                 noise_variance = channel.calc_noise_variance(ebn0, code_rate * bits_per_symbol)
#                 data = channel.add_noise(data, ebn0, code_rate * bits_per_symbol)
#
#             if demodulator is not None:
#                 if demodulator.mode == 'hard':
#                     data = demodulator.demodulate_hard(data)
#                 elif demodulator.mode == 'soft':
#                     data = demodulator.demodulate_soft(data, noise_variance)
#                 else:
#                     raise ValueError("Некорректный режим работы демодулятора. Может быть 'hard' или 'soft'")
#
#             if decoder is not None:
#                 if demodulator.mode == 'hard':
#                     data = decoder.decode(data)
#                 elif demodulator.mode == 'soft':
#                     data = decoder.decode(bits=None, llrs=data)
#                 else:
#                     raise ValueError("Некорректный режим работы демодулятора. Может быть 'hard' или 'soft'")
#
#             bits_processed += params.bits_process_per_iteration
#             bit_errors += count_bit_errors(message_bits, data)
#
#             print("ebn0 = %d, bits_processed = %d, errs = %d, appr_BER = %.7f"
#                   % (ebn0, bits_processed, bit_errors, (bit_errors / bits_processed)))
#
#         ber = bit_errors / bits_processed
#         ber_points.append(ber)
#
#     return BERComputationResult(ber_points.copy(), name)


# qam4_mod = QAMModulator(bits_per_symbol=2, constellation=default_qam_constellations.get_qam_constellation[2])
# awgnc = AWGNChannel()
# qam4_hard_demod = QAMDemodulator.from_qam_modulator(qam4_mod)
# qam4_soft_demod = QAMDemodulator.from_qam_modulator(qam4_mod, mode='soft')
# coder_7_5 = ConvolutionalCoder(np.array([3]), g_matrix=np.array([[7, 5]]))
# decoder_7_5 = ConvolutionalDecoder.from_coder(coder_7_5)
#
# results = []
# p1 = ComputationParameters(2500, 5_000_000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 250_000)
# res = simulate(modulator=qam4_mod, channel=awgnc, demodulator=qam4_hard_demod, name='QAM-4 + hard', params=p1)
# results.append(res)
#
# res = simulate(modulator=qam4_mod, channel=awgnc, demodulator=qam4_soft_demod,
#                coder=coder_7_5, decoder=decoder_7_5, name='QAM-4 + code 7 5 + soft')
# results.append(res)
#
# tcm = TCM()
# res = simulate(modulator=tcm, channel=awgnc, demodulator=tcm, name='TCM')
# results.append(res)
#
# bpsk_hard = bpsk_modem.BPSKModem()
# res = simulate(modulator=bpsk_hard, channel=awgnc, demodulator=bpsk_hard, name='2-psk + hard', params=p1)
# results.append(res)
#
# res = simulate(modulator=bpsk_hard, channel=awgnc, demodulator=bpsk_hard, name='2-psk + code 7 5 + hard',
#                coder=coder_7_5, decoder=decoder_7_5)
# results.append(res)
#
# bpsk_soft = bpsk_modem.BPSKModem(mode='soft')
# res = simulate(modulator=bpsk_soft, channel=awgnc, demodulator=bpsk_soft, name='2-psk + code 7 5 + soft',
#                coder=coder_7_5, decoder=decoder_7_5)
# results.append(res)
#
# plot_ber_computation_results(results)

# modulator = QAMModulator()
# awgnc = AWGNChannel(ebn0_db=0, information_bits_per_symbol=modulator.bits_per_symbol)
# demodulator = QAMDemodulator.from_qam_modulator(modulator)
#
# model = Model([modulator, awgnc, demodulator], name="QAM-16")
#
# params = ComputationParameters(2500, 5_000_000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 250_000)
# res = model.do_modelling(params)
# res.plot()

trellis = cc.Trellis(np.array([3]), g_matrix=np.array([[7, 5]]))
coder_7_5 = ConvCoder(trellis)
decoder_7_5 = ConvDecoder.from_coder(coder_7_5, 'unquantized')
# 'unquantized'

modulator = QAMModulator()
awgnc = AWGNChannel(information_bits_per_symbol=modulator.bits_per_symbol)
demodulator = QAMDemodulator.from_qam_modulator(modulator, mode='soft')

info = Block()
ebn0_db = 10
info.block_output = [[ebn0_db]]

connections = [Connection(coder_7_5, 0, modulator, 0),
               Connection(modulator, 0, awgnc, 0),
               Connection(info, 0, awgnc, 1),
               Connection(awgnc, 0, demodulator, 0),
               Connection(awgnc, 1, demodulator, 1),
               Connection(demodulator, 0, decoder_7_5, 0)]

block_configs = {
    coder_7_5.id: [MethodCallDescription('encode', inputs=[0], outputs=[0])],
    modulator.id: [MethodCallDescription('process', inputs=[0], outputs=[0])],
    awgnc.id: [MethodCallDescription('process', inputs=[0, 1], outputs=[0]),
               MethodCallDescription('calc_noise_variance', inputs=[1], outputs=[1])],
    demodulator.id: [MethodCallDescription('process', inputs=[0, 1], outputs=[0])],
    decoder_7_5.id: [MethodCallDescription('decode', inputs=[0], outputs=[0])]
}

model = Model(blocks=[info, modulator, awgnc, demodulator, coder_7_5, decoder_7_5],
              starting_blocks=[coder_7_5],
              final_block=decoder_7_5,
              connections=connections,
              block_configs=block_configs)

data = np.random.randint(low=0, high=2, size=10_000)
output = model.process(data)
decoded_bits = output[0].pop()
print("ebn0_db = %i, bit_errs = %i" % (ebn0_db, count_bit_errors(data, decoded_bits)))
