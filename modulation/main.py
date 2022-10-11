import numpy as np
import matplotlib.pyplot as plt
import time

from numpy import sign

from channel import AWGNChannel
from qam_demodulation import QAMDemodulator
from qam_modulation import QAMModulator
import default_qam_constellations
import scipy.special as special
import sk_dsp_comm.digitalcom as dc
import sk_dsp_comm.fec_conv as fec

symbols_num = 1_000_000


def gen_bits(seed, bits_num):
    np.random.seed(0)
    return np.random.randint(low=0, high=2, size=bits_num)


def count_bit_errors(input_bits, output_bits):
    return np.count_nonzero(input_bits - output_bits)


def bit_error_rate(input_bits, output_bits):
    errs = count_bit_errors(input_bits, output_bits)
    print("bit errors: " + str(errs))
    res = errs / len(input_bits)
    print("ber: " + str(res))
    return res


def calc_ber_curve(input_bits, bits_per_symbol):
    t = time.time()
    BER_points = []

    qam_modulator = QAMModulator(bits_per_symbol, bit_mapping=None)
    awgn_channel = AWGNChannel()
    qam_demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)

    mod_signal = qam_modulator.modulate(input_bits)

    print("mod_time: %s sec", (time.time() - t))

    max_Eb_N0_dB = 20
    for Eb_N0_dB in range(0, max_Eb_N0_dB):
        t = time.time()

        dirty_sig = awgn_channel.add_noise(mod_signal, Eb_N0_dB, qam_modulator.bits_per_symbol)

        demod_bits = qam_demodulator.demodulate(dirty_sig)

        ber = bit_error_rate(input_bits, demod_bits)
        BER_points.append(ber)
        print("iter_time: %s sec", (time.time() - t))
    return BER_points


def calc_ber(bits_per_symbol, Eb_N0_dB):
    bit_errors = 0

    qam_modulator = QAMModulator(bits_per_symbol, bit_mapping=None)
    awgn_channel = AWGNChannel()
    qam_demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)

    bit_pack_size = 300_000
    bits_processed = 0
    k = 0
    approx_volume = 1_000_000_000
    while (bit_errors < 1000 and bits_processed < 1_000_000_000 and bits_processed < approx_volume):
        np.random.seed(k)
        k += 1
        bits = np.random.randint(low=0, high=2, size=bit_pack_size)

        mod_signal = qam_modulator.modulate(bits)
        dirty_sig = awgn_channel.add_noise(mod_signal, Eb_N0_dB, qam_modulator.bits_per_symbol)
        demod_bits = qam_demodulator.demodulate(dirty_sig)

        bits_processed += bit_pack_size
        bit_errors += count_bit_errors(bits, demod_bits)
        if bits_processed % 2_500_000 == 0:
            approx_ber = bit_errors / bits_processed
            print("approx_ber:", approx_ber)

            if approx_ber > 0:
                s = np.format_float_scientific(approx_ber)
                a, b, c = s.partition("e-")
                # print(s.partition("e-"))
                # print(a, 10 ** int(c))
                # print(100 / float(a), 10 ** int(c))
                approx_volume = (100 / float(a)) * (10 ** int(c))
                print("approx_vol:", approx_volume)
        if bits_processed % 25_000_000 == 0:
            print("bits processed:", (bits_processed))

    ber = bit_errors / bits_processed
    print("Eb_N0_dB=" + str(Eb_N0_dB), "ber=" + str(ber), "k=" + str(k), "bits_processed=" + str(bits_processed))
    print("------------------------------------")
    return ber


def calc_ber_curve_balanced_work(bits_per_symbol):
    bers = []
    border = 2e-5
    Eb_N0_db = 0
    while 1:
        ber = calc_ber(bits_per_symbol, Eb_N0_db)
        bers.append(ber)
        Eb_N0_db += 1

        if ber < border:
            break
    return bers


def plot_ber_curve(bits_per_symbol_list):
    results = []
    for bits_per_symbol in bits_per_symbol_list:
        bits = gen_bits(1, int(symbols_num * bits_per_symbol))
        results.append((bits_per_symbol, calc_ber_curve(bits, bits_per_symbol)))
    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")
    for (bits_per_symbol, BER_points) in results:
        plt.plot(BER_points, '--o', label=str(2 ** bits_per_symbol) + '-QAM')
        plt.legend()
    plt.show()


# bers = calc_ber_curve_balanced_work(bits_per_symbol=1)
#
# plt.yscale("log")
# plt.grid(visible='true')
# plt.xlabel("Eb/N0, dB")
# plt.ylabel("BER")
# plt.plot(bers, '--o', label="balanced_work")

# plot_ber_curve([1])

def calc_ber_curve_for_different_bit_mappings(bits_per_symbol, bit_mappings, mapping_names):
    BER_curves = []
    for bit_mapping in bit_mappings:
        qam_modulator = QAMModulator(bits_per_symbol, bit_mapping)
        awgn_channel = AWGNChannel()
        qam_demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)

        bits = gen_bits(1, int(symbols_num * bits_per_symbol))
        mod_signal = qam_modulator.modulate(bits)

        max_Eb_N0_dB = 20
        BER_points = []
        for Eb_N0_dB in range(0, max_Eb_N0_dB):
            dirty_sig = awgn_channel.add_noise(mod_signal, Eb_N0_dB, qam_modulator.bits_per_symbol)
            demod_bits = qam_demodulator.demodulate(dirty_sig)

            ber = bit_error_rate(bits, demod_bits)
            BER_points.append(ber)
        BER_curves.append(list.copy(BER_points))
        BER_points.clear()


    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")
    i = 0
    for i in range(len(bit_mappings)):
        plt.plot(BER_curves[i], '--o', label=mapping_names[i])
        i += 1
        plt.legend()
    plt.show()


# calc_ber_curve_for_different_bit_mappings(4, [
#     None,
#     [13, 9, 1, 5,
#      12, 8, 0, 4,
#      14, 10, 2, 6,
#      15, 11, 3, 7],
#     [5, 11, 7, 3,
#      1, 0, 9, 4,
#      2, 6, 15, 8,
#      14, 12, 10, 13]
# ], ["default", "gray", "random"])

qam_modulator = QAMModulator(4, bit_mapping=None)
awgn_channel = AWGNChannel()
qam_demodulator = QAMDemodulator.from_qam_modulator(qam_modulator)

import commpy.channelcoding.convcode as cc

def BER_calc(a, b):
    num_ber = np.sum(np.abs(a - b))
    ber = np.mean(np.abs(a - b))
    return int(num_ber), ber

message_bits = np.random.randint(0, 2, size=40_000)

# constraint length
K = 3

# Trellis structure
trellis = cc.Trellis(np.array([K-1]), np.array([[5, 7]]))

constraint = int(np.log2(trellis.number_states)) + 1

# code has rate 1/2 so approximate traceback depth is
tb_depth = 5 * (K - 1)

coded_bits = cc.conv_encode(message_bits, trellis) # encoding

print(len(coded_bits))
modulated = qam_modulator.modulate(coded_bits)

EbNo = 0

noisy = awgn_channel.add_noise(modulated, EbNo, qam_modulator.bits_per_symbol)

demodulated_hard = qam_demodulator.demodulate(noisy)

decoded_hard = cc.viterbi_decode(demodulated_hard, trellis, tb_depth, decoding_type='hard') # decoding (hard decision)
print(len(decoded_hard))

NumErr, BER_hard = BER_calc(message_bits, decoded_hard[:message_bits.size]) # bit-error ratio (hard decision)

print(np.array_equal(message_bits, decoded_hard[:message_bits.size]))
print("conv coding", NumErr, BER_hard)

modulated_no_coding = qam_modulator.modulate(message_bits)
noisy = awgn_channel.add_noise(modulated_no_coding, EbNo, qam_modulator.bits_per_symbol)
demodulated_no_coding = qam_demodulator.demodulate(noisy)

NumErr, BER_hard = BER_calc(message_bits, demodulated_no_coding)
print("no coding", NumErr, BER_hard)
