import numpy as np
import neurokit2 as nk
def signal_sine_noiser_generator (input_signal,sine_freq,scale = 1,shift = 0):
    samples = np.arange(np.shape(input_signal)[0])
    signall = np.sin(samples * 2 * 3.14 * sine_freq)
    signall = signall*scale
    signall = signall+shift
    return input_signal+signall

def signal_white_noise_generator(input_signal,scale = 1,shift = 0):
    samples = (np.shape(input_signal)[0])
    mean = 0
    std = 1
    signall = np.random.normal(mean, std, size=samples)
    signall = signall * scale
    signall = signall + shift
    return input_signal + signall

def signal_mio_noise_generator(input_signal,scale = 1,shift = 0):
    samples = (np.shape(input_signal)[0])
    signall = nk.emg_simulate(length=samples, burst_number=4, burst_duration=0.7)
    signall = signall * scale
    signall = signall + shift
    return input_signal + signall