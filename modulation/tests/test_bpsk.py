import bpsk_modem
import numpy as np
import commpy.channelcoding.convcode as cc
import channel
import main

modem = bpsk_modem.BPSKModem()

def test_modulation():
    input_bits = np.random.randint(low=0, high=2, size=10_000)
    symbols = modem.modulate(input_bits)
    expected = 2 * input_bits - 1

    assert np.array_equal(expected, symbols)


def test_hard_demodulation():
    input_bits = np.random.randint(low=0, high=2, size=10_000)
    symbols = modem.modulate(input_bits)
    demod_bits = modem.demodulate(symbols, demod_type='hard')

    assert np.array_equal(input_bits, demod_bits)


# def test_soft_demodulation():
#     input_bits = np.random.randint(low=0, high=2, size=50_000)
#     K = 3
#     trellis = cc.Trellis(np.array([K - 1]), g_matrix=np.array([[5, 7]]), polynomial_format='Matlab')
#     code_rate = trellis.k / trellis.n
#
#     coded_bits = cc.conv_encode(input_bits, trellis)
#     symbols = modem.modulate(coded_bits)
#
#     ebn0_db = 5
#     awgn_channel = channel.AWGNChannel()
#     noised = awgn_channel.add_noise(symbols, ebn0_db, 1, code_rate)
#
#     demod_soft = modem.demodulate(noised, demod_type='soft',
#                                   noise_var=channel.calc_noise_variance(ebn0_db, 1, code_rate))
#     decoded_bits = cc.viterbi_decode(demod_soft, trellis, decoding_type='unquantized')
#
#     bit_errors, ber = main.ber_calc(input_bits, decoded_bits[:len(input_bits)])
#     print(bit_errors)
