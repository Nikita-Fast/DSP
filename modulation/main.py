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


def count_bit_errors(input_bits, output_bits):
    return np.count_nonzero(input_bits - output_bits)


def bit_error_rate(input_bits, output_bits):
    errs = count_bit_errors(input_bits, output_bits)
    print("bit errors: " + str(errs))
    res = errs / len(input_bits)
    print("ber: " + str(res))
    return res


def calc_ber_curve(input_bits, order):
    t = time.time()
    BER_points = []

    qam_modulator = modulation.QAMModulator(order=order)
    awgn_channel = channel.AWGNChannel()
    qam_demodulator = demodulation.QAMDemodulator(order=order, constellation_points=qam_modulator.qam_symbols())

    # qam_modulator.plot_constellation_points()

    mod_signal = qam_modulator.modulate(input_bits)

    print("mod_time: %s sec", (time.time() - t))

    max_Eb_N0_dB = 14
    for Eb_N0_dB in range(0, max_Eb_N0_dB):
        t = time.time()

        dirty_sig = awgn_channel.add_noise(mod_signal, Eb_N0_dB, qam_modulator.bits_per_symbol)

        demod_bits = qam_demodulator.demodulate(dirty_sig)

        ber = bit_error_rate(input_bits, demod_bits)
        BER_points.append(ber)
        print("iter_time: %s sec", (time.time() - t))
    return BER_points


def calc_ber(order, Eb_N0_dB):
    bit_errors = 0
    qam_modulator = modulation.QAMModulator(order=order)
    awgn_channel = channel.AWGNChannel()
    qam_demodulator = demodulation.QAMDemodulator(order=order, constellation_points=qam_modulator.qam_symbols())

    bit_pack_size = 100_000
    bits_processed = 0
    k = 0
    while (bit_errors < 1000 and bits_processed < 1_000_000_000):
        np.random.seed(k)
        k += 1
        bits = np.random.randint(low=0, high=2, size=bit_pack_size)

        mod_signal = qam_modulator.modulate(bits)
        dirty_sig = awgn_channel.add_noise(mod_signal, Eb_N0_dB, qam_modulator.bits_per_symbol)
        demod_bits = qam_demodulator.demodulate(dirty_sig)

        bits_processed += bit_pack_size
        bit_errors += count_bit_errors(bits, demod_bits)
        # print(bit_errors, bits_processed)
    ber = bit_errors / bits_processed
    print("Eb_N0_dB="+str(Eb_N0_dB), "ber="+str(ber), "k="+str(k), "bits_processed="+str(bits_processed))
    return ber


def calc_ber_curve_balanced_work(order):
    bers = []
    border = 1e-7
    Eb_N0_db = 0
    while 1:
        ber = calc_ber(order, Eb_N0_db)
        bers.append(ber)
        Eb_N0_db += 1

        if ber < border:
            break
    return bers

def plot_ber_curve(orders):
    results = []
    for order in orders:
        bits = gen_bits(1, int(symbols_num * np.log2(order)))
        results.append((order, calc_ber_curve(bits, order)))
    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")
    for (order, BER_points) in results:
        plt.plot(BER_points, '--o', label=str(order) + '-QAM')
        plt.legend()
    plt.show()


bers = calc_ber_curve_balanced_work(4)
plt.yscale("log")
plt.grid(visible='true')
plt.xlabel("Eb/N0, dB")
plt.ylabel("BER")
plt.plot(bers, '--o', label="balanced_work")

plot_ber_curve([4])
