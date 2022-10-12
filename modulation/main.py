import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import time

import qam_modulation
from channel import AWGNChannel
from qam_demodulation import QAMDemodulator
from qam_modulation import QAMModulator
import commpy.channelcoding.convcode as cc
from modulation_type import ModulationType

symbols_num = 1_0_000


def gen_bits(seed, bits_num: int):
    np.random.seed(0)
    return np.random.randint(low=0, high=2, size=bits_num)


def count_bit_errors(input_bits, output_bits):
    return np.sum(np.abs(input_bits - output_bits), dtype=int)


def ber_calc(a, b):
    bit_errors = np.sum(np.abs(a - b), dtype=int)
    ber = np.mean(np.abs(a - b))
    return bit_errors, ber


def calc_ber_curve_fixed_volume(symbols_num: int, bits_per_symbol_ls: List[int], max_ebn0: int):
    ber_curves = []
    ber_curves_descs = []
    ber_points = []

    for mod_type in ModulationType:
        if mod_type == ModulationType.QAM:
            for bits_per_symbol in bits_per_symbol_ls:
                qam_modulator = QAMModulator(bits_per_symbol, bit_mapping=None)
                awgn_channel = AWGNChannel()
                qam_demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)

                in_bits = gen_bits(bits_per_symbol, int(symbols_num * qam_modulator.bits_per_symbol))

                desc = (mod_type.name + "-" + str(2 ** qam_modulator.bits_per_symbol))

                for ebn0 in range(max_ebn0 + 1):
                    modulated = qam_modulator.modulate(in_bits)
                    noised = awgn_channel.add_noise(modulated, ebn0, qam_modulator.bits_per_symbol)
                    demodulated = qam_demodulator.demodulate(noised)

                    _, ber = ber_calc(in_bits, demodulated)
                    ber_points.append(ber)

                    progress = str(int(100 * ebn0 / max_ebn0)) + "%"
                    print("", end='\r', flush=True)
                    print("Modelling of " + desc + " progress: " + progress, end='', flush=False)

                ber_curves.append(ber_points.copy())
                ber_points.clear()

                ber_curves_descs.append(desc)
                print("")
    return ber_curves, ber_curves_descs


def calc_ber_curve_balanced_volume(max_volume: int, qam_modulator: QAMModulator, max_ebn0: int):
    ber_curves = []
    ber_curves_descs = []
    ber_points = []
    name = "QAM-" + str(2 ** qam_modulator.bits_per_symbol)

    awgn_channel = AWGNChannel()
    qam_demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)

    bit_pack_size = 100_000 * qam_modulator.bits_per_symbol

    print("Computing exact BER for " + name)
    for ebn0 in range(max_ebn0 + 1):
        bit_errors = 0
        bits_processed = 0
        approx_volume = max_volume

        while (bit_errors < 1000 and bits_processed < max_volume and bits_processed < approx_volume):
            bits = np.random.randint(low=0, high=2, size=bit_pack_size)

            mod_signal = qam_modulator.modulate(bits)
            dirty_sig = awgn_channel.add_noise(mod_signal, ebn0, qam_modulator.bits_per_symbol)
            demod_bits = qam_demodulator.demodulate(dirty_sig)

            bits_processed += bit_pack_size
            bit_errors += count_bit_errors(bits, demod_bits)

            if bits_processed % (10 * bit_pack_size) == 0:
                approx_ber = bit_errors / bits_processed

                if approx_ber > 0:
                    s = np.format_float_scientific(approx_ber)
                    a, b, c = s.partition("e-")
                    approx_volume = (100 / float(a)) * (10 ** int(c))
            print("", end='\r', flush=True)
            print("\tebn0 = %d, bits_processed = %d, errs = %d, appr_BER = %.10f"
                  % (ebn0, bits_processed, bit_errors, (bit_errors / bits_processed)), end='', flush=False)

        ber = bit_errors / bits_processed
        ber_points.append(ber)

    ber_curves.append(ber_points.copy())
    ber_points.clear()
    ber_curves_descs.append(name + " balanced volume")
    return ber_curves, ber_curves_descs


def calc_ber_curve_for_different_bit_mappings(bits_per_symbol: int, bit_mappings, mapping_names: List[str],
                                              max_ebn0: int, symbols_num: int):
    ber_curves = []
    ber_curves_descs = []
    ber_points = []
    for bit_mapping, mapping_name in zip(bit_mappings, mapping_names):
        qam_modulator = QAMModulator(bits_per_symbol, bit_mapping)
        awgn_channel = AWGNChannel()
        qam_demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)

        bits = gen_bits(1, int(symbols_num * qam_modulator.bits_per_symbol))
        mod_signal = qam_modulator.modulate(bits)

        for ebn0 in range(0, max_ebn0 + 1):
            dirty_sig = awgn_channel.add_noise(mod_signal, ebn0, qam_modulator.bits_per_symbol)
            demod_bits = qam_demodulator.demodulate(dirty_sig)

            _, ber = ber_calc(bits, demod_bits)
            ber_points.append(ber)
        ber_curves.append(ber_points.copy())
        ber_points.clear()
        ber_curves_descs.append("QAM-" + str(2 ** qam_modulator.bits_per_symbol) + " " + mapping_name + " mapping")
    return ber_curves, ber_curves_descs


def calc_ber_curve_convenc(K: int, g_matrix, qam_modulator: qam_modulation.QAMModulator, max_ebn0: int,
                           symbols_num: int):
    in_bits = gen_bits(1, symbols_num * qam_modulator.bits_per_symbol)
    awgn_channel = AWGNChannel()
    qam_demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)

    trellis = cc.Trellis(np.array([K - 1]), g_matrix)
    tb_depth = 5 * (K - 1)

    ber_curves = []
    ber_curves_descs = []
    ber_points = []

    for use_conv_coding in {False, True}:
        t = time.time()
        for ebn0 in range(max_ebn0 + 1):
            bits = in_bits

            if use_conv_coding:
                bits = cc.conv_encode(in_bits, trellis)

            modulated = qam_modulator.modulate(bits)
            noised = awgn_channel.add_noise(modulated, ebn0, qam_modulator.bits_per_symbol)
            demodulated = qam_demodulator.demodulate(noised)
            out_bits = demodulated

            if use_conv_coding:
                out_bits = cc.viterbi_decode(demodulated, trellis, tb_depth, decoding_type='hard')

            _, ber = ber_calc(in_bits, out_bits[:len(in_bits)])
            ber_points.append(ber)

            print("", end='\r', flush=True)
            print(("conv_enc" if use_conv_coding else "no_enc") + " ebn0 = %d, ber = %.10f done" % (ebn0, ber), end='',
                  flush=False)

        ber_curves.append(ber_points.copy())
        ber_points.clear()

        desc = ("QAM-" + str(2 ** qam_modulator.bits_per_symbol) + " " + ("conv_enc" if use_conv_coding else "no_enc"))
        ber_curves_descs.append(desc)
        print("\n" + desc + " elapsed_time: %.2f" % (time.time() - t))

    return ber_curves, ber_curves_descs


def plot_ber_curves(ber_curves, ber_curves_descs):
    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")

    for ber_curve, desc in zip(ber_curves, ber_curves_descs):
        plt.plot(ber_curve, '--o', label=desc)
        plt.legend()
    plt.show()


# # тест всех видов модуляции на одинаковом объеме символов
# ber_curves, ber_curves_descs = calc_ber_curve_fixed_volume(symbols_num, bits_per_symbol_ls=[1,2,3,4,5], max_ebn0=20)
# plot_ber_curves(ber_curves, ber_curves_descs)
#
# высчитываем BER по схеме 1000 ошибок vs превышение объема
# modulator = QAMModulator(bits_per_symbol=2, bit_mapping=None)
# ber_curves, ber_curves_descs = \
#     calc_ber_curve_balanced_volume(max_volume=50_000_000, qam_modulator=modulator, max_ebn0=15)
# plot_ber_curves(ber_curves, ber_curves_descs)
#
# тестируем влияние битовой раскладки
# ber_curves, ber_curves_descs = calc_ber_curve_for_different_bit_mappings(4, [
#     None,
#     [0, 1, 2, 3, 15, 14, 13, 12, 7, 6, 5, 4, 8, 9, 10, 11],
#     qam_modulation.gray_codes(bits_per_symbol=4)
# ], ["default", "random", "gray"], max_ebn0=15, symbols_num=1_000_000)
# plot_ber_curves(ber_curves, ber_curves_descs)

# тестируем влияние свёрточного кодера
qam_modulator = QAMModulator(bits_per_symbol=4, bit_mapping=None)
ber_curves, ber_curves_descs = calc_ber_curve_convenc(K=3, g_matrix=np.array([[5, 7]]), qam_modulator=qam_modulator,
                                                      max_ebn0=14, symbols_num=100_000)
plot_ber_curves(ber_curves, ber_curves_descs)
