import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse as sps

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

def bits_to_lil_matrix(bits):
    length = len(bits) // 4
    m = sps.lil_matrix((length, 16), dtype=bool)
    j = 0
    for i in range(length):
        b0 = bits[4 * i + 3]
        b1 = bits[4 * i + 2]
        b2 = bits[4 * i + 1]
        b3 = bits[4 * i]
        v = (b3 << 3) + (b2 << 2) + (b1 << 1) + b0
        m[j, v] = 1
        j += 1
    return m

def modulate_using_matrix(bits):
    lil = bits_to_lil_matrix(bits)
    m = lil.tocoo()

    c_vector = constellation_points[:, None]

    t = time.time()
    res = m.dot(c_vector)
    print("vector product: %s sec", (time.time() - t))

    return res

def bits_to_nums(bits):
    length = len(bits) // 4
    output = np.empty(symbols_num, dtype=complex)
    j = 0
    for i in range(length):
        b0 = bits[4 * i + 3]
        b1 = bits[4 * i + 2]
        b2 = bits[4 * i + 1]
        b3 = bits[4 * i]
        output[j] = (b3 << 3) + (b2 << 2) + (b1 << 1) + b0
        j += 1
    return output

def mod(bits):
    # t = time.time()
    length = len(bits) // 4
    j = 0
    output = np.empty(symbols_num, dtype=complex)

    # print(bits)
    # ls = [(0 if (i % 8 < 4) else bits[i - (4 * (i // 8 + 1))]) for i in range(0, 2 * len(bits))]
    # print("%s" %(time.time() - t))
    # a = np.array(ls)
    # vals = np.packbits(a)
    # return np.vectorize(gray_mapping.get)(vals)

    for i in range(length):
        b0 = bits[4 * i + 3]
        b1 = bits[4 * i + 2]
        b2 = bits[4 * i + 1]
        b3 = bits[4 * i]
        key = (b3 << 3) + (b2 << 2) + (b1 << 1) + b0
        output[j] = gray_mapping[key]
        j += 1
    # print("%s" % (time.time() - t))
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
    return np.sqrt(var)*np.random.randn(symbols_num) + 1j*np.sqrt(var)*np.random.randn(symbols_num)


def ints_to_bits(values):
    bits = np.unpackbits(values.astype(np.uint8))

    # строка выше для каждого числа создаёт 8 бит, а нужны только 4. Поэтому первые четыре из каждых
    # восьми необходимо удалить
    mask = np.tile(np.r_[np.zeros(4, int), np.ones(4, int)], len(values))
    target_bits = bits[np.nonzero(mask)]

    return target_bits


def bit_error_rate(input_bits, output_bits):
    errs = np.count_nonzero(input_bits - output_bits)
    print("bit errors: " + str(errs))
    return errs / len(input_bits)


def plot_ber_curve(input_bits):
    t = time.time()
    BERs = []
    # input_signal = mod(input_bits)
    input_signal = (modulate_using_matrix(input_bits)).ravel()

    signal_power = power(input_signal)
    print("mod_time: %s sec", (time.time() - t))

    for EbN0_dB in range(0, 17):
        t = time.time()
        P_noise = calc_noise_power(EbN0_dB, signal_power)
        n = awgn(symbols_num, P_noise / 2, 0)

        dirty_sig = input_signal + n

        indices = np.abs(dirty_sig[:, None] - constellation_points[None, :]).argmin(axis=1)

        received_bits = ints_to_bits(indices)

        ber = bit_error_rate(input_bits, received_bits)
        BERs.append(ber)
        print("iter_time: %s sec", (time.time() - t))
    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")
    plt.plot(np.linspace(start=0, stop=16, num=17), BERs, '--bo', label='16-QAM')
    plt.legend()
    plt.show()


bits = gen_bits(0, bits_num)
plot_ber_curve(bits)

# test = np.random.randint(0, 16, 1_000_000)
#
# for j in range(0, 5):
#     t = time.time()
#     b = np.vectorize(gray_mapping.get)(test)
#     print("mod: %s sec", (time.time() - t))






