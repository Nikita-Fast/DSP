import numpy as np
import matplotlib.pyplot as plt

############# 4-ary Quadrature Amplitude Modulation ###################

EbN0_dB = 10
bits_num = 1_000_000
bits_per_symbol = 2
constellation_points = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
bits_constellation_points = [0b00,0b01,0b10,0b11]
symbols_num = bits_num // bits_per_symbol
seed = 0

def gen_bits(seed, bits_num):
    np.random.seed(0)
    return np.random.randint(low=0, high=2, size=bits_num)


bits = gen_bits(0, bits_num)


def map_bits(bits):
    x = 2 * bits[0] + 1 * bits[1]
    if x == 0b00:
        return 1 + 1j
    elif x == 0b01:
        return -1 + 1j
    elif x == 0b10:
        return -1 - 1j
    elif x == 0b11:
        return 1 - 1j


# signal = [map_bits(bits[2 * i: 2 * i + 2]) for i in range(symbols_num)]

def power(sig):
    abs = np.abs(sig)
    return ((abs ** 2).sum()) / len(sig)

def calc_noise_power(EbN0_dB, signal):
    EsN0_dB = EbN0_dB + 10 * np.log10(bits_per_symbol)
    SNR_dB = EsN0_dB
    SNR = 10 ** (SNR_dB / 10)
    P_sig = power(signal)
    P_noise = P_sig / SNR
    return P_noise

def awgn(symbols_num, var, seed):
    np.random.seed(seed)
    return np.random.normal(loc=0.0, scale=np.sqrt(var), size=symbols_num) + \
    1j * np.random.normal(loc=0.0, scale=np.sqrt(var), size=symbols_num)

def distance(p1, p2):
    return np.sqrt(
        (np.real(p1) - np.real(p2)) ** 2 + (np.imag(p1) - np.imag(p2)) ** 2
    )


def guess(p):
    dists = list(map(lambda x: distance(x, p), constellation_points))
    m = min(dists)
    i = dists.index(m)
    return constellation_points[i]

demodulate = np.vectorize(guess)

def symbol_error_rate(x, y):
    return np.count_nonzero(x-y) / len(x)


def symbol_to_bits(s):
    i = constellation_points.index(s)
    return bits_constellation_points[i]


def symbols_to_bits(symbols):
    bits = np.zeros(len(symbols) * bits_per_symbol,int)
    j = 0
    for i in range(len(symbols)):
        s = symbols[i]
        v = bits_constellation_points[constellation_points.index(s)]
        b0 = v & 0b01
        b1 = (v & 0b10) >> 1
        bits[j] = b1
        j+=1
        bits[j] = b0
        j+=1
    return bits

def bit_error_rate(input_bits, output_bits):
    return np.count_nonzero(input_bits-output_bits) / len(input_bits)


def plot_ber_curve(input_bits):
    BERs = []
    input_signal = [map_bits(input_bits[2 * i: 2 * i + 2]) for i in range(symbols_num)]
    for EbN0_dB in range(1,11):
        P_noise = calc_noise_power(EbN0_dB, input_signal)
        n = awgn(symbols_num, P_noise / 2, 0)
        dirty_sig = input_signal + n

        demodulated_signal = demodulate(dirty_sig)
        received_bits = symbols_to_bits(demodulated_signal)
        ber = bit_error_rate(input_bits, received_bits)
        BERs.append(ber)
    plt.yscale("log")
    plt.plot([1,2,3,4,5,6,7,8,9,10], BERs)
    plt.show()

plot_ber_curve(bits)

# P_noise = calc_noise_power(EbN0_dB, signal)
# n = awgn(symbols_num, P_noise / 2, 0)
# dirty_sig = signal + n

# x_max = np.max(np.abs(np.real(dirty_sig)))
# y_max = np.max(np.abs(np.imag(dirty_sig)))
# plt.axis([-1.05*x_max, 1.05*x_max, -1.05*y_max, 1.05*y_max])
# plt.scatter(np.real(dirty_sig), np.imag(dirty_sig))
# plt.show()


# def distance(p1, p2):
#     return np.sqrt(
#         (np.real(p1) - np.real(p2)) ** 2 + (np.imag(p1) - np.imag(p2)) ** 2
#     )
#
#
# def guess(p):
#     dists = list(map(lambda x: distance(x, p), constellation_points))
#     m = min(dists)
#     i = dists.index(m)
#     return constellation_points[i]


# demodulate = np.vectorize(guess)
# demodulated_signal = demodulate(dirty_sig)


# def symbol_error_rate(x, y):
#     return np.count_nonzero(x-y) / len(x)
#
# print("SER =",symbol_error_rate(signal, demodulated_signal))
#
# def symbol_to_bits(s):
#     i = constellation_points.index(s)
#     return bits_constellation_points[i]
#
#
# def symbols_to_bits(symbols):
#     bits = np.zeros(len(symbols) * bits_per_symbol,int)
#     j = 0
#     for i in range(len(symbols)):
#         s = symbols[i]
#         v = bits_constellation_points[constellation_points.index(s)]
#         b0 = v & 0b01
#         b1 = (v & 0b10) >> 1
#         bits[j] = b1
#         j+=1
#         bits[j] = b0
#         j+=1
#     return bits
#
#
# received_bits = symbols_to_bits(demodulated_signal)
#
# def bit_error_rate(input_bits, output_bits):
#     return np.count_nonzero(input_bits-output_bits) / len(input_bits)
#
# print("BER =", bit_error_rate(bits, received_bits))
