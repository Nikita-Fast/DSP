import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.spatial.distance as dst

bits_num = 4_000_000
bits_per_symbol = 4
constellation_points = np.array([1 + 1j, 1 + 3j, 1 - 1j, 1 - 3j, 3 + 1j, 3 + 3j, 3 - 1j, 3 - 3j, -1 + 1j, -1 + 3j, -1 - 1j,
                        -1 - 3j, -3 + 1j, -3 + 3j, -3 - 1j, -3 - 3j])
symbols_num = bits_num // bits_per_symbol
seed = 0


def gen_bits(seed, bits_num):
    np.random.seed(0)
    return np.random.randint(low=0, high=2, size=bits_num)


gray_mapping = {
    0: 1 + 1j,
    1: 1 + 3j,
    2: 1 - 1j,
    3: 1 - 3j,
    4: 3 + 1j,
    5: 3 + 3j,
    6: 3 - 1j,
    7: 3 - 3j,
    8: -1 + 1j,
    9: -1 + 3j,
    10: -1 - 1j,
    11: -1 - 3j,
    12: -3 + 1j,
    13: -3 + 3j,
    14: -3 - 1j,
    15: -3 - 3j
}


def mod(bits):
    length = len(bits) // 4
    j = 0
    output = np.empty(symbols_num, dtype=complex)
    for i in range(length):
        b0 = bits[4 * i + 3]
        b1 = bits[4 * i + 2]
        b2 = bits[4 * i + 1]
        b3 = bits[4 * i]
        key = (b3 << 3) + (b2 << 2) + (b1 << 1) + b0
        output[j] = gray_mapping[key]
        j += 1
    return output

def power(sig):
    return (((np.abs(sig)) ** 2).sum()) / len(sig)


def calc_noise_power(EbN0_dB, signal_power):
    EsN0_dB = EbN0_dB + 10 * np.log10(bits_per_symbol)
    SNR_dB = EsN0_dB
    SNR = 10 ** (SNR_dB / 10)
    P_noise = signal_power / SNR
    return P_noise


def awgn(symbols_num, var, seed):
    np.random.seed(seed)
    return np.random.normal(loc=0.0, scale=np.sqrt(var), size=symbols_num) + \
           1j * np.random.normal(loc=0.0, scale=np.sqrt(var), size=symbols_num)


def ints_to_bits(values):
    bits = np.unpackbits(values.astype(np.uint8))

    # строка выше для каждого числа создаёт 8 бит, а нужны только 4. Поэтому первые четыре из каждых
    # восьми необходимо удалить
    mask = np.tile(np.r_[np.zeros(4, int), np.ones(4, int)], len(values))
    target_bits = bits[np.nonzero(mask)]

    return target_bits


def bit_error_rate(input_bits, output_bits):
    return np.count_nonzero(input_bits - output_bits) / len(input_bits)


def plot_ber_curve(input_bits):
    total_time = time.time()
    prepare_time = time.time()
    BERs = []
    input_signal = mod(input_bits)
    signal_power = power(input_signal)

    print("--- prepare %s ---" % (time.time() - prepare_time))
    print("++++++++++++++++")
    for EbN0_dB in range(0, 17):
        iter_time = time.time()
        n_time = time.time()

        P_noise = calc_noise_power(EbN0_dB, signal_power)
        n = awgn(symbols_num, P_noise / 2, 0)
        n_time = time.time() - n_time

        dirty_sig = input_signal + n

        demod_time = time.time()
        indices = np.abs(dirty_sig[:, None] - constellation_points[None, :]).argmin(axis=1)
        demod_time = time.time() - demod_time

        bits_time = time.time()
        received_bits = ints_to_bits(indices)
        bits_time = time.time() - bits_time

        ber_time = time.time()
        ber = bit_error_rate(input_bits, received_bits)
        ber_time = time.time() - ber_time
        BERs.append(ber)
        iter_time = time.time() - iter_time

        print("--- iter %s ---" % (iter_time))
        print("--- noise %s ---" % (n_time / iter_time))
        print("--- demod %s ---" % (demod_time / iter_time))
        print("--- bits %s ---" % (bits_time / iter_time))
        print("--- ber %s ---" % (ber_time / iter_time))
        print("------------")
    print("--- total_time %s ---" % (time.time() - total_time))
    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")
    plt.plot(np.linspace(start=0, stop=16, num=17), BERs, '--bo', label='16-QAM')
    plt.legend()
    plt.show()


bits = gen_bits(0, bits_num)
plot_ber_curve(bits)

