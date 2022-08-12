import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.signal as sg

def fft_filtering(sg, h):
    length = len(sg)
    sg_fft = np.fft.fft(sg, length)
    h_fft = np.fft.fft(h, length)
    mul_fft = np.multiply(sg_fft, h_fft)
    return np.fft.ifft(mul_fft)

# f_c = 100

f_s = 100_000
def psd(who):
    w, f = np.fft.fftshift(sg.welch(who, fs=f_s, return_onesided=False, detrend=False))
    plt.plot(f, 10 * np.log10(w))
def my_psd(who):
    w, f = np.fft.fftshift(sg.welch(who, fs=f_s, return_onesided=False, detrend=False))
    return w, f

local_f_c = 10_000
time = np.linspace(start=0,stop=1, num=f_s)
samples = 20000

# x = np.random.default_rng().normal(loc=0.0, scale=0.5, size=samples)
h = np.genfromtxt('h.csv', delimiter=',')
# x = fft_filtering(x, h)
# # x = np.convolve(x,h,'same')
# psd(x)
# plt.show()
#
# carrier = np.cos(2 * np.pi * local_f_c * time[0:samples])
# y= x*carrier
# fig, ax = plt.subplots()
# ax.plot(x, label='input')
# ax.plot(y, label='modulated')
# ax.legend()
# plt.show()
#
# u = y * 2 * np.cos(2 * np.pi * local_f_c * time[0:samples])
# psd(u)
# plt.show()
#
h1 = np.genfromtxt('h1.csv', delimiter=',')
# # x_demod = fft_filtering(u, h1)
# x_demod = np.convolve(u, h1,'same')
#
# _, ax = plt.subplots()
# ax.plot(x, label='input')
# ax.plot(x_demod, label='demod')
# ax.legend()
# plt.show()
#
# ################## QAM-part ###############################
# x = np.random.default_rng().normal(loc=0.0, scale=0.5, size=samples)
# x_i = np.convolve(x,h,'same')
# x = np.random.default_rng().normal(loc=0.0, scale=0.5, size=samples)
# x_q = np.convolve(x,h,'same')
#
# y = x_i * np.cos(2*np.pi*local_f_c*time[0:samples]) + x_q * np.sin(2*np.pi*local_f_c*time[0:samples])
# psd(y)
# plt.show()
#
# u_i = y * 2 * np.cos(2*np.pi*local_f_c*time[0:samples])
# u_q = y * 2 * np.sin(2*np.pi*local_f_c*time[0:samples])
#
# x_i_demod = np.convolve(u_i, h1, 'same')
# x_q_demod = np.convolve(u_q, h1, 'same')
#
# _, ax = plt.subplots()
# ax.plot(x_i, label='x_i')
# ax.plot(x_i_demod, label='x_i demod')
# ax.plot(x_q, label='x_q')
# ax.plot(x_q_demod, label='x_q demod')
# ax.legend()
# plt.show()
#
# #######Complex Quadrature Amplitude Modulation###########
# x = np.random.default_rng().normal(loc=0.0, scale=0.5, size=samples)
# x_i = np.convolve(x,h,'same')
# x = np.random.default_rng().normal(loc=0.0, scale=0.5, size=samples)
# x_q = np.convolve(x,h,'same')
#
# x = x_i + 1j*x_q
# y = np.real(x * np.exp(-1j*2*np.pi*local_f_c*time[0:samples]))
#
# u = y * 2 * np.exp(1j*2*np.pi*local_f_c*time[0:samples])
# x_demod = np.convolve(u, h1, 'same')
# x_i_demod = np.real(x_demod)
# x_q_demod = np.imag(x_demod)
#
# _, ax = plt.subplots()
# ax.plot(x_i, label='x_i input')
# ax.plot(x_q, label='x_q input')
# ax.plot(x_i_demod, label='x_i demod')
# ax.plot(x_q_demod, label='x_q demod')
# ax.legend()
# plt.show()



################## Additive White Gaussian Noise ####################

# x = np.random.default_rng().normal(loc=0.0, scale=0.5, size=samples)
# x_i = np.convolve(x,h,'same')
# x = np.random.default_rng().normal(loc=0.0, scale=0.5, size=samples)
# x_q = np.convolve(x,h,'same')
#
# x = x_i + 1j*x_q
# # print(np.std(x_i))
# # print(np.std(x_q))
# n = np.random.default_rng().normal(loc=0.0, scale=0.025, size=samples)
#
# y = np.real(x * np.exp(-1j*2*np.pi*local_f_c*time[0:samples])) + n
#
# u = y * 2 * np.exp(1j*2*np.pi*local_f_c*time[0:samples])
# x_demod = np.convolve(u, h1, 'same')
# x_i_demod = np.real(x_demod)
# x_q_demod = np.imag(x_demod)
#
# _, ax = plt.subplots(2)
# ax[0].set_title('Modulation/demodulation with AWGN')
# ax[0].plot(x_i, label='x_i input')
# ax[0].plot(x_i_demod, label='x_i demod')
# ax[1].plot(x_q, label='x_q input')
# ax[1].plot(x_q_demod, label='x_q demod')
# ax[0].legend()
# ax[1].legend()
# plt.show()
#
# _, ax = plt.subplots(2)
# ax[0].set_title('PSD of input/demodulated')
# w, f = my_psd(x_i)
# ax[0].plot(f, 10 * np.log10(w), label='input x_i')
# w, f = my_psd(x_i_demod)
# ax[0].plot(f, 10 * np.log10(w), label='demodulated x_i')
#
# w, f = my_psd(x_q)
# ax[1].plot(f, 10 * np.log10(w), label='input x_q')
# w, f = my_psd(x_q_demod)
# ax[1].plot(f, 10 * np.log10(w), label='demodulated x_q')
#
# ax[0].legend()
# ax[1].legend()
# plt.show()


############# 4-ary Quadrature Amplitude Modulation ###################

f_b = 20000
bits_per_symbol = 2
f_symb = f_b // bits_per_symbol
samples_per_symbol = f_s // f_symb
symbols = samples // samples_per_symbol
print("symbols: " + str(symbols))
bits = symbols * 2
np.random.seed(0)
b = np.random.randint(low=0,high=2,size=bits)

b_1 = b[::2]
b_2 = b[1::2]

sn = -2 *(b_1-0.5) + 1j*-2*(b_2-0.5)
st = [(0 if (i % samples_per_symbol != 0) else sn[i // samples_per_symbol] ) for i in range(len(sn) * samples_per_symbol)]
st = np.concatenate((np.zeros(samples_per_symbol // 2),st),axis=0)

h = np.genfromtxt('impls_interpol.csv', delimiter=',')
x = np.convolve(st, h,'same')

_, ax = plt.subplots(2)
ax[0].set_title('real part')
ax[0].plot(np.real(st), label='discrete')
ax[0].plot(np.real(x), label='analog')

ax[1].set_title('imag part')
ax[1].plot(np.imag(st), label='discrete')
ax[1].plot(np.imag(x), label='analog')

ax[0].legend()
ax[1].legend()
plt.show()
#### вся предыдущая часть это генерация аналогового сигнала из потока битов

y = np.real(x[0:samples] * np.exp(-1j*2*np.pi*local_f_c*time[0:samples]))

#вычисляем мощность сигнала
#то, что тут получается мне совершенно не нравится
def power(sig):
    abs = np.abs(sig)
    return ((abs**2).sum()) / len(sig)

SNRdb = 3
gamma = 10**(SNRdb/10)
P = power(y)
E_b = P * (1 / f_symb)
N_0 = P / gamma
print(gamma, P, E_b, N_0, 10*np.log10(E_b / N_0))

n = np.random.default_rng().normal(loc=0.0, scale=np.sqrt(N_0 / 2), size=samples)

psd(y)
psd(n)
plt.show()

y += n

u = y * 2 * np.exp(1j*2*np.pi*local_f_c*time[0:samples])

x_demod = np.convolve(u, h1, 'same')

_, ax = plt.subplots(2)
ax[0].set_title('real part')
ax[0].plot(np.real(x), label='x')
ax[0].plot(np.real(x_demod), label='x_demod')

ax[1].set_title('imag part')
ax[1].plot(np.imag(x), label='x')
ax[1].plot(np.imag(x_demod), label='x_demod')

ax[0].legend()
ax[1].legend()
plt.show()

sampled = x_demod[(samples_per_symbol // 2)::samples_per_symbol]
b_1_hat = (np.real(sampled) < 0) * 1
b_2_hat = (np.imag(sampled) < 0) * 1
b_hat = [(b_1_hat[i//2] if i % 2 == 0 else b_2_hat[i//2]) for i in range(2 * len(b_1_hat))]

bit_errors_num = (b ^ b_hat).sum()
print("BER = " + str(bit_errors_num / bits))





# symbs = 6
# # bits = np.random.randint(0, high=2, size=symbs*4)
# bits = [1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#
#
# def get_amplitude(a, b):
#     if a == 0:
#         if b == 0:
#             return -0.821
#         else:
#             return -0.22
#     else:
#         if b == 0:
#             return 0.22
#         else:
#             return 0.821
#
#
# def modulate(bit_input):
#     output = np.empty(shape=[0])
#
#     length = len(bit_input)
#     symbols = length // 4
#     for i in range(symbols):
#         q0 = bit_input[i * 4 + 0]
#         q1 = bit_input[i * 4 + 1]
#         i0 = bit_input[i * 4 + 2]
#         i1 = bit_input[i * 4 + 3]
#         q_ampl = get_amplitude(q0, q1)
#         i_ampl = get_amplitude(i0, i1)
#         T = 1 / f_c
#         time = np.linspace(start=i * T, stop=(i + 1) * T, num=50)
#         discrete_symbol = q_ampl * np.cos(2 * np.pi * f_c * time) + i_ampl * np.sin(2 * np.pi * f_c * time)
#
#         output = np.concatenate((output, discrete_symbol), axis=0)
#     return output
#
#
# modulated_sig = modulate(bits)
# cos = np.cos(2*np.pi*f_c*np.linspace(start=0, stop=(1 / f_c) *symbs, num=len(modulated_sig)))
# sin = np.sin(2*np.pi*f_c*np.linspace(start=0, stop=(1 / f_c) *symbs, num=len(modulated_sig)))
# r_i = modulated_sig * cos
# r_q = modulated_sig * sin
# plt.plot(modulated_sig)
# plt.plot(r_i)
# plt.plot(r_q,color='red')
# plt.show()
#
# # хотим отфильтровать гармоники с частотой 2 * f_c
#
# h = np.genfromtxt('h_qam.csv', delimiter=',')
# i = fft_filtering(r_i, h)
# q = fft_filtering(r_q, h)
#
# plt.plot(i)
# plt.plot(q, color='green')
# plt.show()