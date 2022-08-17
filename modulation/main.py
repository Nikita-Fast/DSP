import numpy as np
import matplotlib.pyplot as plt
import time

bits_num = 4_000_000
bits_per_symbol = 4
constellation_points = [1 + 1j, 1 + 3j, 1 - 1j, 1 - 3j, 3 + 1j, 3 + 3j, 3 - 1j, 3 - 3j, -1 + 1j, -1 + 3j, -1 - 1j,
                        -1 - 3j, -3 + 1j, -3 + 3j, -3 - 1j, -3 - 3j]
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

symbols_to_dec = {
    1 + 1j : 0,
    1 + 3j : 1,
     1 - 1j : 2,
     1 - 3j : 3,
    3 + 1j : 4,
    3 + 3j:5,
    3 - 1j:6,
    3 - 3j:7,
    -1 + 1j:8,
    -1 + 3j:9,
     -1 - 1j:10,
     -1 - 3j:11,
     -3 + 1j:12,
     -3 + 3j:13,
     -3 - 1j:14,
     -3 - 3j:15
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
    abs = np.abs(sig)
    return ((abs ** 2).sum()) / len(sig)


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


def distance(p1, p2):
    return np.abs(p1-p2)


def guess(p):
    points = []
    offset = 0
    if (np.real(p) >= 0):
        if (np.imag(p) >= 0):
            points = [constellation_points[0],constellation_points[1],
                      constellation_points[4],constellation_points[5]]
        else:
            points = [constellation_points[2], constellation_points[3],
                      constellation_points[6], constellation_points[7]]
            offset = 2
    elif np.imag(p) >= 0:
        points = [constellation_points[8], constellation_points[9],
                  constellation_points[12], constellation_points[13]]
        offset = 8
    else:
        points = [constellation_points[10], constellation_points[11],
                  constellation_points[14], constellation_points[15]]
        offset = 10
    dists = list(map(lambda x: distance(x, p), points))
    m = min(dists)
    bonus = 0
    i = dists.index(m)
    if i >= 2:
        bonus = 2

    return gray_mapping[offset + i + bonus]



demodulate = np.vectorize(guess)


def symbol_error_rate(x, y):
    return np.count_nonzero(x - y) / len(x)


def symbols_to_bits(symbols):
    bits = np.zeros(len(symbols) * bits_per_symbol, int)
    j = 0
    for i in range(len(symbols)):
        s = symbols[i]
        v = symbols_to_dec[s]
        b0 = v & 0b01
        b1 = (v & 0b10) >> 1
        b2 = (v & 0b100) >> 2
        b3 = (v & 0b1000) >> 3
        bits[j] = b3
        j += 1
        bits[j] = b2
        j += 1
        bits[j] = b1
        j += 1
        bits[j] = b0
        j += 1
    return bits


def bit_error_rate(input_bits, output_bits):
    return np.count_nonzero(input_bits - output_bits) / len(input_bits)


def plot_ber_curve(input_bits):
    start_time = time.time()
    BERs = []
    input_signal = mod(input_bits)
    signal_power = power(input_signal)
    for EbN0_dB in range(0, 17):
        P_noise = calc_noise_power(EbN0_dB, signal_power)
        n = awgn(symbols_num, P_noise / 2, 0)
        dirty_sig = input_signal + n

        demodulated_signal = demodulate(dirty_sig)

        received_bits = symbols_to_bits(demodulated_signal)
        ber = bit_error_rate(input_bits, received_bits)
        BERs.append(ber)
        print("--- %s seconds ---" % (time.time() - start_time))
    plt.yscale("log")
    plt.grid(visible='true')
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")
    plt.plot(np.linspace(start=0, stop=16, num=17), BERs, '--bo', label='16-QAM')
    plt.legend()
    plt.show()


bits = gen_bits(0, bits_num)
plot_ber_curve(bits)

