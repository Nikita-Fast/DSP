import numpy as np
import matplotlib.pyplot as plt
import time

import channel
import modulation
import demodulation

symbols_num = 1_000_000


def gen_bits(seed, bits_num):
    np.random.seed(0)
    return np.random.randint(low=0, high=2, size=bits_num)


def bit_error_rate(input_bits, output_bits):
    errs = np.count_nonzero(input_bits - output_bits)
    print("bit errors: " + str(errs))
    return errs / len(input_bits)


def calc_ber_curve(input_bits, order):
    t = time.time()
    BER_points = []

    qam_modulator = modulation.QAMModulator(order=order)
    awgn_channel = channel.AWGNChannel()
    qam_demodulator = demodulation.QAMDemodulator(order=order, constellation_points=qam_modulator.qam_symbols())

    # qam_modulator.plot_constellation_points()

    mod_signal = qam_modulator.modulate(input_bits)

    print("mod_time: %s sec", (time.time() - t))

    max_Eb_N0_dB = 30
    for Eb_N0_dB in range(0, max_Eb_N0_dB):
        t = time.time()

        dirty_sig = awgn_channel.add_noise(mod_signal, Eb_N0_dB, qam_modulator.bits_per_symbol)

        demod_bits = qam_demodulator.demodulate(dirty_sig)

        ber = bit_error_rate(input_bits, demod_bits)
        BER_points.append(ber)
        print("iter_time: %s sec", (time.time() - t))
    return BER_points


def plot_ber_curve(orders):
    results = []
    for order in orders:
        bits = gen_bits(0, int(symbols_num * np.log2(order)))
        results.append((order, calc_ber_curve(bits, order)))
    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")
    for (order, BER_points) in results:
        plt.plot(BER_points, '--o', label=str(order) + '-QAM')
        plt.legend()
    plt.show()


plot_ber_curve([4, 16, 32])
