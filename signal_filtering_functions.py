from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def signal_lopass_filtration(input_signal,fs,order = 2,cutoff_freq = 1,plot = True,filter ='butter'):
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

        signal_filtered = signal.filtfilt(b,a,input_signal)

        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        t_vector = list(range(np.shape(input_signal)[0]))
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
        plt.xlim(freq_vector[stop_pas_point[0][-1]],fs*0.5)
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
        plt.xlim(freq_vector[stop_pas_point[0][-1]],fs*0.5)
        plt.xlabel('Frequency [Hz]')
        plt.subplots_adjust(hspace=0.3)
        plt.show(block=False)
        #--------------------#

        signal_filtered = signal.filtfilt(b,a,input_signal)

        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        t_vector = list(range(np.shape(input_signal)[0]))
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


def signal_bandpass_filtration(input_signal,fs,order = 2,cutoff_freq = [1,2],plot = True,filter ='butter'):
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
        plt.title("Highpass Filter Frequency Response")
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

        signal_filtered = signal.filtfilt(b,a,input_signal)

        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        t_vector = list(range(np.shape(input_signal)[0]))
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