import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sg

f_s = 1000
start = 0
stop = 1
length = (stop - start) * f_s
t = np.linspace(start, stop, length)

noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=length)

# plt.figure()
# plt.hist(noise, bins=100)
# plt.show()

f1 = 8
f2 = 55
f3 = 155
clean_signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f3 * t)
signal = clean_signal + noise


def rms(arr):
    s = 0.0
    for i in range(len(arr)):
        s += arr[i] ** 2
    return np.sqrt(s / len(arr))

#вычисляем отношение сигнал шум
snr_db = 20*np.log10(rms(clean_signal) / rms(noise))
print("SNR in dB: ", snr_db)

def psd(who):
    w, f = np.fft.fftshift(sg.welch(who, fs=f_s, return_onesided=False, detrend=False))
    plt.plot(f, 10 * np.log10(w))


# моя реализация фильтра через свёртку, у которой нет искажений на стыке пакетов входного сигнала
# Параметры фильтра:
# N = 128, f_s = 1000 Hz, F_pb = 25, F_sb = 50
h = np.genfromtxt('imp_response.csv', delimiter=',')
flt_length = len(h)
delay_line = np.zeros(flt_length - 1)


def get_x(samples, index):
    if index < 0:
        return delay_line[index + flt_length - 1]
    else:
        return samples[index]


def convolve(imp_resp, pos, packet):
    s = 0
    for j in range(flt_length):
        s += get_x(packet, pos - j) * imp_resp[j]
    return s


def update_delay_line(packet):
    j = len(packet) - 1
    base = flt_length - 2
    for i in range(flt_length - 1):
        delay_line[i] = packet[j - base + i]


def my_filter(packet, imp_resp):
    y = np.zeros(len(packet))
    for i in range(len(packet)):
        y[i] = convolve(imp_resp, i, packet)
    update_delay_line(packet)
    return y


sg_fft = np.fft.fft(signal, length)
h_fft = np.fft.fft(h, length)
mul_fft = np.multiply(sg_fft, h_fft)
filtered_fft = np.fft.ifft(mul_fft)

filtered_conv = np.concatenate((np.convolve(signal[:length // 2], h),
                          np.convolve(signal[length // 2:], h)))

zi = sg.lfilter_zi(h, 1)
y1, zf = sg.lfilter(h, 1, signal[:length // 2], zi=zi * signal[0])
y2, _ = sg.lfilter(h, 1, signal[length // 2:], zi=zf)
filtered_scipy = np.append(y1, y2)

# psd(signal)
# psd(filtered_fft)
psd(h)
psd(clean_signal)
psd(noise)
plt.show()

# plt.figure()
# plt.plot(noise)
# plt.show()


plt.figure()
plt.plot(signal)
# plt.plot(filtered_conv)
plt.plot(np.concatenate(
    (my_filter(signal[0:153],h),
     my_filter(signal[153:476],h),
     my_filter(signal[476:length],h))
), color='red')
#plt.plot(filtered_scipy, color='black')
plt.plot(filtered_fft, color='green')
plt.show()

#интерполяция
int_signal = np.sin(2 * np.pi * 160 * t)
# plt.figure()
# plt.plot(int_signal)
# plt.show()
F_s = 4000
a = np.zeros(F_s)
k = F_s // f_s
for i in range(length):
    a[k*i] = int_signal[i]

h = np.genfromtxt('interpolation.csv', delimiter=',')
flt_length = len(h)
delay_line = np.zeros(flt_length - 1)

b = my_filter(a,h)

plt.figure()
plt.plot(a)
plt.plot(b)
plt.show()