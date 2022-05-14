import numpy as np
import neurokit2 as nk

def signal_noise_ratio(signal,noise):
    rms_signal = np.sqrt(np.mean(signal**2))
    rms_noise = np.sqrt(np.mean(noise ** 2))
    snr = 20*np.log10(rms_signal/rms_noise)
    return snr

def signal_sine_noiser_generator (input_signal,sample_rate,sine_freq,scale = 1,shift = 0,log = False):
    start_time = 0
    end_time = np.ceil(np.shape(input_signal)[0]/sample_rate)
    time = np.arange(start_time, end_time, 1 / sample_rate)
    signall = np.sin(2 * np.pi * sine_freq * time)
    signall = signall*scale
    signall = signall+shift
    snr = signal_noise_ratio(input_signal,signall)
    if log == True:
        print('#---------------------Sine_noise_generator-------------------------#')
        print('Signal_to_noise_ratio = '+str(snr)+' dB')
        print('Sine_freq = '+str(sine_freq)+ ' Hz')
        print('Scale = ' + str(scale))
        print('Shift = ' + str(shift))
        print('#------------------------------------------------------------------#')
    return snr,signall, input_signal+signall

def signal_white_noise_generator(input_signal,scale = 1,shift = 0,log = False):
    samples = (np.shape(input_signal)[0])
    mean = 0
    std = 1
    signall = np.random.normal(mean, std, size=samples)
    signall = signall * scale
    signall = signall + shift
    snr = signal_noise_ratio(input_signal, signall)
    if log == True:
        print('#---------------------White_noise_generator-------------------------#')
        print('Signal_to_noise_ratio = ' + str(snr) + ' dB')
        print('Std = ' + str(std))
        print('Mean = ' + str(mean))
        print('Scale = ' + str(scale))
        print('Shift = ' + str(shift))
        print('#------------------------------------------------------------------#')
    return snr,signall, input_signal + signall

def signal_mio_noise_generator(input_signal,fs = 1000,scale = 1,shift = 0,log = False):
    samples = (np.shape(input_signal)[0])
    samp_rate = None
    if fs:
        samp_rate = fs
    signall = nk.emg_simulate(length=samples, burst_number=4, burst_duration=0.7,sampling_rate=samp_rate)
    signall = signall * scale
    signall = signall + shift
    snr = signal_noise_ratio(input_signal, signall)
    if log==True:
        print('#---------------------Mio_noise_generator-------------------------#')
        print('Signal_to_noise_ratio = ' + str(snr) + ' dB')
        print('Scale = ' + str(scale))
        print('Shift = ' + str(shift))
        print('#------------------------------------------------------------------#')
    return snr,signall, input_signal + signall