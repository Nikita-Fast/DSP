import numpy as np
import matplotlib.pyplot as plt

EbN0_dB = 10
bits_num = 4000000
bits_per_symbol = 4
constellation_points = [1 + 1j,1 + 3j,1 - 1j,1 - 3j,3 + 1j,3 + 3j,3 - 1j,3 - 3j,-1 + 1j, -1 + 3j, -1 - 1j,-1 - 3j,-3 + 1j,-3 + 3j,-3 - 1j,-3 - 3j]
bits_constellation_points = [0b0000,0b0001,0b0010,0b0011,0b0100,0b0101,0b0110,0b0111,0b1000,0b1001,0b1010,0b1011,0b1100,0b1101,0b1110,0b1111]
symbols_num = bits_num // bits_per_symbol
seed = 0

def gen_bits(seed, bits_num):
    np.random.seed(0)
    return np.random.randint(low=0, high=2, size=bits_num)


bits = gen_bits(0, bits_num)


def map_bits(bits):
    x = 8 * bits[0] + 4 * bits[1] + 2 * bits[2] + 1 * bits[3]
    if x == 0b0000:
        return 1 + 1j
    elif x == 0b0001:
        return 1 + 3j
    elif x == 0b0010:
        return 1 - 1j
    elif x == 0b0011:
        return 1 - 3j
    elif x == 0b0100:
        return 3 + 1j
    elif x == 0b0101:
        return 3 + 3j
    elif x == 0b0110:
        return 3 - 1j
    elif x == 0b0111:
        return 3 - 3j
    elif x == 0b1000:
        return -1 + 1j
    elif x == 0b1001:
        return -1 + 3j
    elif x == 0b1010:
        return -1 - 1j
    elif x == 0b1011:
        return -1 - 3j
    elif x == 0b1100:
        return -3 + 1j
    elif x == 0b1101:
        return -3 + 3j
    elif x == 0b1110:
        return -3 - 1j
    elif x == 0b1111:
        return -3 - 3j


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
        b2 = (v & 0b100) >> 2
        b3 = (v & 0b1000) >> 3
        bits[j] = b3
        j += 1
        bits[j] = b2
        j += 1
        bits[j] = b1
        j+=1
        bits[j] = b0
        j+=1
    return bits

def bit_error_rate(input_bits, output_bits):
    return np.count_nonzero(input_bits-output_bits) / len(input_bits)


def plot_ber_curve(input_bits):
    BERs = []
    # input_signal = [map_bits(input_bits[2 * i: 2 * i + 2]) for i in range(symbols_num)]
    input_signal = [map_bits(input_bits[4 * i: 4 * i + 4]) for i in range(symbols_num)]
    for EbN0_dB in range(0,17):
        P_noise = calc_noise_power(EbN0_dB, input_signal)
        n = awgn(symbols_num, P_noise / 2, 0)
        dirty_sig = input_signal + n

        demodulated_signal = demodulate(dirty_sig)
        print(demodulated_signal[0:5])
        received_bits = symbols_to_bits(demodulated_signal)
        ber = bit_error_rate(input_bits, received_bits)
        BERs.append(ber)
    plt.yscale("log")
    plt.grid(visible='true')
    # plt.plot([1,2,3,4,5,6,7,8,9,10], BERs)
    plt.xlabel("Eb/N0, dB")
    plt.ylabel("BER")
    plt.plot(np.linspace(start=0,stop=16,num=17), BERs,'--bo', label='16-QAM')
    # plt.plot(range(10), '--bo', label='line with marker')
    plt.legend()
    plt.show()

plot_ber_curve(bits)
