from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.fftpack import rfft, irfft, fftfreq,rfftfreq


def impz(b, a):
    # Define the impulse sequence of length 60
    length = 3000
    impulse = np.repeat(0., length)
    impulse[0] = 1.
    x = np.arange(0, length)

    # Compute the impulse response
    response = signal.lfilter(b, a, impulse)

    # Plot filter impulse and step response:
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.stem(x, response, 'm', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Impulse response', fontsize=15)

    plt.subplot(212)
    step_line = np.ones(length)
    step = signal.lfilter(b, a, step_line)
    #step = np.cumsum(response)


    # Compute step response of the system
    plt.stem(x, step, 'g', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Step response', fontsize=15)
    plt.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.show()

def signal_lopass_filtration(input_signal,fs,order = 2,cutoff_freq = 1,plot = True,filter ='butter',imz_flag = False):
    start_time = time.time()
    def butter_lowpass(cutoff, fs, order=5,):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def firwin_lowpass(cutoff,fs,order = 3):
        if order%2 == 0:
            order = order+1
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b= signal.firwin(order, normal_cutoff)
        a = 1
        return b,a
    if filter =='butter':
        b, a = butter_lowpass(cutoff_freq, fs, order=order)

    if filter == 'firwin':
        b,a = firwin_lowpass(cutoff_freq, fs, order=order)

    if imz_flag == True:
        impz(b, a)
    signal_filtered = signal.filtfilt(b, a, input_signal)
    print("--- %s seconds ---" % (time.time() - start_time))
    if plot:
        # -----Freq and phase resp----#
        plt.figure()
        w, h = signal.freqz(b, a, worN=8000)
        freq_vector = 0.5 * fs * w / np.pi
        resp_in_dB = 20 * np.log10(np.abs(h))
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(freq_vector,resp_in_dB, 'b')
        stop_pas_point = np.where(resp_in_dB < -100)
        if filter == 'butter':
            plt.plot(cutoff_freq, -3, 'ko')
        plt.axvline(cutoff_freq, color='k')
        plt.xlim(0, freq_vector[stop_pas_point[0][0]])
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.ylim(-100,10)
        plt.grid()

        plt.subplot(212,sharex = ax1)
        angles = np.unwrap(np.angle(h))
        angles[stop_pas_point] = np.nan
        plt.plot(freq_vector, np.degrees(angles), 'g')
        plt.ylabel('Angle (deg)', color='g')
        plt.grid()
        plt.xlim(0, freq_vector[stop_pas_point[0][0]])
        plt.xlabel('Frequency [Hz]')
        plt.subplots_adjust(hspace=0.3)
        plt.show(block=False)
        #--------------------#

        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        t_vector = list(range(np.shape(input_signal)[0]))
        t_vector = np.array(t_vector) / fs
        plt.plot(t_vector, input_signal, 'b')
        plt.plot(t_vector, signal_filtered, 'r',linewidth=0.5)
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='b')
        low_lim = np.min(input_signal)
        high_lim = np.max(input_signal)
        delta = (high_lim-low_lim)*0.05
        plt.ylim(low_lim-delta,high_lim+delta)
        plt.title("Butterworth low pass filter filtration result")
        plt.grid()
        plt.subplot(212,sharex = ax1)
        plt.plot(t_vector, signal_filtered, 'r')
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='r')
        plt.ylim(low_lim - delta, high_lim + delta)
        plt.grid()
        plt.tight_layout()
        plt.show()
    return signal_filtered


def signal_highpass_filtration(input_signal,fs,order = 2,cutoff_freq = 1,plot = True,filter ='butter'):
    def butter_highpass(cutoff, fs, order=5,):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def firwin_highpass(cutoff,fs,order = 3):
        if order%2 == 0:
            order = order+1
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b= signal.firwin(order, normal_cutoff, pass_zero=False)
        a = 1
        return b,a

    if filter == 'butter':
        b, a = butter_highpass(cutoff_freq, fs, order=order)

    if filter == 'firwin':
        b,a = firwin_highpass(cutoff_freq, fs, order=order)

    if plot:
        # -----Freq and phase resp----#
        plt.figure()
        w, h = signal.freqz(b, a, worN=8000)
        freq_vector = 0.5 * fs * w / np.pi
        resp_in_dB = 20 * np.log10(np.abs(h))
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(freq_vector,resp_in_dB, 'b')
        stop_pas_point = np.where(resp_in_dB < -100)
        if filter == 'butter':
            plt.plot(cutoff_freq, -3, 'ko')
        plt.axvline(cutoff_freq, color='k')
        #plt.xlim(freq_vector[stop_pas_point[0][-1]],fs*0.5)
        plt.title("Highpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.ylim(-100,10)
        plt.grid()

        plt.subplot(212,sharex = ax1)
        angles = np.unwrap(np.angle(h))
        angles[stop_pas_point] = np.nan
        plt.plot(freq_vector, np.degrees(angles), 'g')
        plt.ylabel('Angle (deg)', color='g')
        plt.grid()
        #plt.xlim(freq_vector[stop_pas_point[0][-1]],fs*0.5)
        plt.xlabel('Frequency [Hz]')
        plt.subplots_adjust(hspace=0.3)
        plt.show(block=False)
        #--------------------#

        signal_filtered = signal.filtfilt(b,a,input_signal)

        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        t_vector = list(range(np.shape(input_signal)[0]))
        t_vector = np.array(t_vector) / fs
        plt.plot(t_vector, input_signal, 'b')
        plt.plot(t_vector, signal_filtered, 'r',linewidth=0.5)
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='b')
        low_lim = np.min(input_signal)
        high_lim = np.max(input_signal)
        delta = (high_lim-low_lim)*0.05
        plt.ylim(low_lim-delta,high_lim+delta)
        plt.title("Butterworth low pass filter filtration result")
        plt.grid()
        plt.subplot(212,sharex = ax1)
        plt.plot(t_vector, signal_filtered, 'r')
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='r')
        plt.ylim(low_lim - delta, high_lim + delta)
        plt.grid()
        plt.tight_layout()
        plt.show()
    return signal_filtered


def signal_bandpass_filtration(input_signal,fs,order = 2,cutoff_freq = [1,2],plot = True,filter ='butter',imz_flag = False):
    start_time = time.time()
    cutoff_freq = np.array(cutoff_freq)
    def butter_bandpass(cutoff, fs, order=5,):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='bandpass', analog=False)
        return b, a

    def firwin_bandpass(cutoff,fs,order = 3):
        if order%2 == 0:
            order = order+1
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b= signal.firwin(order, normal_cutoff, pass_zero=False)
        a = 1
        return b,a

    if filter == 'butter':
        b, a = butter_bandpass(cutoff_freq, fs, order=order)

    if filter == 'firwin':
        b,a = firwin_bandpass(cutoff_freq, fs, order=order)

    if imz_flag == True:
        impz(b, a)

    signal_filtered = signal.filtfilt(b, a, input_signal)
    print("--- %s seconds ---" % (time.time() - start_time))
    if plot:
        # -----Freq and phase resp----#
        plt.figure()
        w, h = signal.freqz(b, a, worN=8000)
        freq_vector = 0.5 * fs * w / np.pi
        resp_in_dB = 20 * np.log10(np.abs(h))
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(freq_vector,resp_in_dB, 'b')
        stop_pas_point = np.where(resp_in_dB > -100)
        if filter == 'butter':
            plt.plot(cutoff_freq[0], -3, 'ko')
            plt.plot(cutoff_freq[1], -3, 'ko')
        plt.axvline(cutoff_freq[0], color='k')
        plt.axvline(cutoff_freq[1], color='k')

        plt.xlim(freq_vector[stop_pas_point[0][0]],freq_vector[stop_pas_point[0][-1]])
        plt.title("Bandpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.ylim(-100,10)
        plt.grid()

        plt.subplot(212,sharex = ax1)
        angles = np.unwrap(np.angle(h))
        #stop_pas_point = np.where(resp_in_dB < -100)

        #angles[stop_pas_point] = np.nan
        plt.plot(freq_vector, np.degrees(angles), 'g')
        plt.ylabel('Angle (deg)', color='g')
        plt.grid()

        plt.xlim(freq_vector[stop_pas_point[0][0]],freq_vector[stop_pas_point[0][-1]])
        plt.xlabel('Frequency [Hz]')
        plt.subplots_adjust(hspace=0.3)
        plt.show(block=False)
        #--------------------#


        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        t_vector = list(range(np.shape(input_signal)[0]))
        t_vector = np.array(t_vector)/fs
        plt.plot(t_vector, input_signal, 'b')
        plt.plot(t_vector, signal_filtered, 'r',linewidth=0.5)
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='b')
        low_lim = np.min(input_signal)
        high_lim = np.max(input_signal)
        delta = (high_lim-low_lim)*0.05
        plt.ylim(low_lim-delta,high_lim+delta)
        plt.title("Butterworth bandpass filter filtration result")
        plt.grid()
        plt.subplot(212,sharex = ax1)
        plt.plot(t_vector, signal_filtered, 'r')
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='r')
        plt.ylim(low_lim - delta, high_lim + delta)
        plt.grid()
        plt.tight_layout()
        plt.show()
    return signal_filtered

def irnotch_filter(input_signal,fs,quality_factor = 0.005,cutoff_freq = 0.01,plot = True,filter ='butter',imz_flag = False,detrend = True):
    start_time = time.time()
    def irnotchh(cutoff, fs, quality_factor):
        b_notch, a_notch = signal.iirnotch(cutoff, quality_factor, fs)
        return b_notch,a_notch

    b,a = irnotchh(cutoff_freq,fs,quality_factor)
    if imz_flag == True:
        impz(b, a)

    signal_filtered = signal.filtfilt(b, a, input_signal)
    if detrend == True:
        signal_filtered = signal.detrend(signal_filtered)  # deetrend
    print("--- %s seconds ---" % ((time.time() - start_time)))
    if plot:
        # -----Freq and phase resp----#
        plt.figure()
        w, h = signal.freqz(b, a, worN=8000)
        freq_vector = 0.5 * fs * w / np.pi
        resp_in_dB = 20 * np.log10(np.abs(h))
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(freq_vector, resp_in_dB, 'b')
        stop_pas_point = np.where(resp_in_dB > -100)

        plt.xlim(freq_vector[stop_pas_point[0][0]], freq_vector[stop_pas_point[0][-1]])
        plt.title("Notch Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.ylim(-50, 10)
        plt.grid()

        plt.subplot(212, sharex=ax1)
        angles = np.unwrap(np.angle(h))
        # stop_pas_point = np.where(resp_in_dB < -100)

        # angles[stop_pas_point] = np.nan
        plt.plot(freq_vector, np.degrees(angles), 'g')
        plt.ylabel('Angle (deg)', color='g')
        plt.grid()

        plt.xlim(freq_vector[stop_pas_point[0][0]], freq_vector[stop_pas_point[0][-1]])
        plt.xlabel('Frequency [Hz]')
        plt.subplots_adjust(hspace=0.3)
        plt.show(block=False)
        # --------------------#

        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        t_vector = list(range(np.shape(input_signal)[0]))
        t_vector = np.array(t_vector) / fs
        plt.plot(t_vector, input_signal, 'b')
        plt.plot(t_vector, signal_filtered, 'r', linewidth=0.5)
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='b')
        low_lim = np.min(input_signal)
        high_lim = np.max(input_signal)
        delta = (high_lim - low_lim) * 0.05
        plt.ylim(low_lim - delta, high_lim + delta)
        plt.title("Butterworth bandpass filter filtration result")
        plt.grid()
        plt.subplot(212, sharex=ax1)
        plt.plot(t_vector, signal_filtered, 'r')
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='r')
        plt.ylim(low_lim - delta, high_lim + delta)
        plt.grid()
        plt.tight_layout()
        plt.show()
    return signal_filtered

def signal_FFT_filtration(input_signal,fs,cutoff_freq = [1,2],bandstop = False,treshold = None,plot = True):
    start_time = time.time()
    input_signal = np.array(input_signal)
    W = fftfreq(input_signal.size, d=1/fs)
    f_signal = rfft(input_signal)
    # If our original signal time was in seconds, this is now in Hz
    cut_f_signal = f_signal.copy()
    from_freq =  cutoff_freq[0]
    if treshold:
        fft_abs = np.abs(cut_f_signal)
        cut_f_signal[fft_abs <= treshold] = 0

    to_freq = cutoff_freq[1]
    if bandstop:
        cut_f_signal[(W > from_freq * 2)&(W < to_freq * 2)] = 0
    else:
        cut_f_signal[(W <from_freq*2)] = 0
        cut_f_signal[(W > to_freq*2)] = 0


    signal_filtered= irfft(cut_f_signal)
    print("--- %s seconds ---" % (time.time() - start_time))

    if plot:
        N = np.shape(input_signal)[0]

        xf = rfftfreq(N, 1 / fs)

        ylim_max = np.max(np.abs(cut_f_signal))

        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        plt.title('FFT before filtering')
        markerline, stemlines, baseline = plt.stem(xf, np.abs(f_signal), markerfmt=" ")
        plt.ylim(0, ylim_max)
        plt.setp(stemlines, 'linewidth', 0.8)
        plt.xlabel('Frequency in Hertz [Hz]')
        plt.ylabel('Spectrum Magnitude')
        plt.xlim(-5, cutoff_freq[1] + np.round(fs / 7))
        plt.subplot(212, sharex=ax1)
        plt.title('FFT after filtering')
        markerline, stemlines, baseline = plt.stem(xf, np.abs(cut_f_signal), markerfmt=" ")
        plt.setp(stemlines, 'linewidth', 0.8)
        plt.xlim(-5, cutoff_freq[1] + np.round(fs / 7))
        plt.xlabel('Frequency in Hertz [Hz]')
        plt.ylabel('Spectrum Magnitude')
        plt.subplots_adjust(hspace=0.53)
        plt.ylim(0, ylim_max)
        plt.show()

        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        t_vector = list(range(np.shape(input_signal)[0]))
        t_vector = np.array(t_vector) / fs
        plt.plot(t_vector, input_signal, 'b')
        plt.plot(t_vector, signal_filtered, 'r', linewidth=0.5)
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='b')
        low_lim = np.min(input_signal)
        high_lim = np.max(input_signal)
        delta = (high_lim - low_lim) * 0.05
        plt.ylim(low_lim - delta, high_lim + delta)
        plt.title("Butterworth bandpass filter filtration result")
        plt.grid()
        plt.subplot(212, sharex=ax1)
        plt.plot(t_vector, signal_filtered, 'r')
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='r')
        plt.ylim(low_lim - delta, high_lim + delta)
        plt.grid()
        plt.tight_layout()
        plt.show()
    return signal_filtered
