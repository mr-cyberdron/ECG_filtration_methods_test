import numpy as np

import signal_filtering_functions
from ECG_generation_func import ECG_generation
import Plottermaan_lib as pl
import noise_generation_functions as noise
from signal_filtering_functions import signal_bandpass_filtration,signal_highpass_filtration,irnotch_filter
from scipy import signal
import matplotlib.pyplot as plt
import neurokit2 as nk


def ecg_signals_comparsion(sig1, sig2, fs, comparing_peak_num=1, QRS_delta_time=0.8, plot=True):
    sig1 = np.array(sig1)
    sig2 = np.array(sig2)

    signal_dif = sig2 - sig1

    def peak_mass(ecg, fs):
        _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=fs, )
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg, rpeaks, sampling_rate=fs, method="dwt", show=False,
                                                 show_type='all')
        waves_dwt['ECG_R_Peaks'] = list(rpeaks['ECG_R_Peaks'])
        ecg_points = waves_dwt
        return ecg_points

    delta_time = QRS_delta_time  # sec
    samples_in_time = (delta_time / 2) * fs
    sig1_R_peaks = peak_mass(sig1, fs)['ECG_R_Peaks']
    sig2_R_peaks = peak_mass(sig2, fs)['ECG_R_Peaks']
    peak_num = comparing_peak_num
    sig1_peak_value = sig1[sig1_R_peaks[peak_num]]
    sig2_peak_value = sig2[sig2_R_peaks[peak_num]]
    delta = sig2_peak_value - sig1_peak_value
    sig1 = sig1 + delta
    sig1_from_count = int(sig1_R_peaks[peak_num] - samples_in_time)
    sig1_to_count = int(sig1_R_peaks[peak_num] + samples_in_time)
    sig2_from_count = int(sig2_R_peaks[peak_num] - samples_in_time)
    sig2_to_count = int(sig2_R_peaks[peak_num] + samples_in_time)
    sig1_qrs_complex = sig1[sig1_from_count:sig1_to_count]
    sig2_qrs_complex = sig2[sig2_from_count:sig2_to_count]

    if plot == True:

        plt.figure()
        t_vector = list(range(np.shape(sig1)[0]))
        t_vector = np.array(t_vector) / fs
        ax0 = plt.subplot(2, 1, 1)
        plt.plot(t_vector, sig1, 'b')
        plt.plot(t_vector, sig2, 'r', linewidth=0.5)

        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='b')
        low_lim = np.min(sig1)
        high_lim = np.max(sig1)
        delta = (high_lim - low_lim) * 0.05
        plt.ylim(low_lim - delta, high_lim + delta)
        plt.title("QRS complexes comparsion")
        plt.grid()

        ax1 = plt.subplot(212,sharex=ax0)

        plt.plot(t_vector, signal_dif, 'b')
        plt.xlabel('Time [sec]')
        plt.ylabel('Difference', color='b')
        plt.ylim(-1,1)
        plt.grid()
        plt.show(block = False)

        plt.figure()
        qrs_t_vector = list(range(np.shape(sig1_qrs_complex)[0]))
        qrs_t_vector = np.array(qrs_t_vector) / fs
        plt.plot(qrs_t_vector, sig1_qrs_complex, 'b')
        plt.plot(qrs_t_vector, sig2_qrs_complex, 'r', linewidth=0.5)
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude', color='b')
        plt.grid()
        plt.show()

    def euclidean_distance(x, y):
        return np.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    def manhattan_distance(x, y):
        return sum(abs(a - b) for a, b in zip(x, y))

    def square_rooted(x):
        return round(np.sqrt(sum([a * a for a in x])), 3)

    def cosine_similarity(x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = square_rooted(x) * square_rooted(y)
        return round(numerator / float(denominator), 3)

    compare1_part = sig1
    compare2_part = sig2
    # compare1_part =sig1_qrs_complex
    # compare2_part = sig2_qrs_complex

    sig1_2_euclidian_distance = euclidean_distance(compare1_part,compare2_part)
    sig1_2_manhttan_distance = manhattan_distance(compare1_part,compare2_part)
    sig1_2_cosine_similarity = cosine_similarity(compare1_part,compare2_part)

    print('Euclidian_distance: '+str(round(sig1_2_euclidian_distance,4)))
    print('Manhttan_distance: '+str(round(sig1_2_manhttan_distance,4)))
    print('Cosine similarity: '+str(round(sig1_2_cosine_similarity,4)))
    return sig1_2_euclidian_distance, sig1_2_manhttan_distance, sig1_2_cosine_similarity
'''
Generate initial ECG
'''
fs = 1000
ecg_line = ECG_generation(
    duration = 10,
    hr = 80,
    fs = fs,
    P_wave_splitting_flag = True,
    R_wave_splitting_flag = True,
    delta_wave_flag = True,
    f_wave_flag = True,
    extrasystole = True,
    extrasystoly_n_central_points = 3,
    show_steps = False
)
'''
Noising ECG signal
'''
_,isoline_drift_corline,signal_noised = noise.signal_sine_noiser_generator(ecg_line,fs,0.25,scale=0.8,shift=0)
_,white_noise_corline,signal_noised = noise.signal_white_noise_generator(signal_noised,scale=0.02,shift=0)
_,miosignal_corline,signal_noised = noise.signal_mio_noise_generator(signal_noised,fs=fs,scale=0.15,shift=0)

# pl.masplot(ecg_line,isoline_drift_corline,white_noise_corline,miosignal_corline,signal_noised,
#            line_names=['Clean_ecg', 'Isoline drift noise', 'White noise','Mio noise', 'Noised signal'],
#            fs=fs, dotstyle='large',
#            fig_title='Generation of noised signal',
#            line_width=3, y_axis_names=['Amplitude mV', 'Amplitude mV', 'Amplitude mV', 'Amplitude mV', 'Amplitude mV'],
#            x_axis_names=['Time, sec', 'Time, sec', 'Time, sec', 'Time, sec', 'Time, sec']
#            )

"""
Filtration test
"""
print('#--------------------------------ISOLINE DRIFT--------------------------------#')
def isoline_drift_butter(input_ecg,fs,
                         noise_freq = 0.5,noise_scale = 0.8,
                         filter_order = 4, cutoff_freq = [1,200],plot = True,
                         imz_flag =False):
    ecg_line = input_ecg
    # basline drift
    isoline_snr,isoline_drift_corline,signal_noised = noise.signal_sine_noiser_generator(ecg_line,fs,noise_freq,
                                                                                         scale=noise_scale,shift=0)
    #Filtration - bandpass butter
    print('Butter_bandpass')
    isoline_butter_bandpass_filt = signal_bandpass_filtration(signal_noised,
                                                              fs,order=filter_order,
                                                              cutoff_freq=cutoff_freq,
                                                              plot=plot,filter='butter',imz_flag=imz_flag)
    euclid,manhttan,cosine = ecg_signals_comparsion(ecg_line,isoline_butter_bandpass_filt,fs,plot=plot)
    return noise_scale,isoline_snr,euclid,manhttan,cosine

def isoline_drift_firwin(input_ecg,fs,
                         noise_freq = 0.5,noise_scale = 0.8,
                         filter_order = 3000, cutoff_freq = [1,200],plot = True,imz_flag = False):
    ecg_line = input_ecg
    # basline drift
    isoline_snr,isoline_drift_corline,signal_noised = noise.signal_sine_noiser_generator(ecg_line,fs,noise_freq,
                                                                                         scale=noise_scale,shift=0)
    #Filtration - bandpass butter
    print('Firwin_bandpass')

    isoline_butter_bandpass_filt = signal_bandpass_filtration(signal_noised,
                                                              fs,order=filter_order,
                                                              cutoff_freq=cutoff_freq,
                                                              plot=plot,filter='firwin',imz_flag=imz_flag)
    euclid,manhttan,cosine = ecg_signals_comparsion(ecg_line,isoline_butter_bandpass_filt,fs,plot=plot)
    return noise_scale,isoline_snr,euclid,manhttan,cosine

def like_2_staged_baseline_filter(input_ecg,fs,
                         noise_freq = 0.5,noise_scale = 0.8,
                         quality_factor = 0.005, cutoff_freq = 50,plot = True,imz_flag = False):
    ecg_line = input_ecg
    # basline drift
    isoline_snr, isoline_drift_corline, signal_noised = noise.signal_sine_noiser_generator(ecg_line, fs, noise_freq,
                                                                                           scale=noise_scale, shift=0)
    # Filtration - bandpass butter
    print('Firwin_bandpass')

    isoline_butter_bandpass_filt = irnotch_filter(signal_noised,
                                                              fs, quality_factor=quality_factor,
                                                              cutoff_freq=cutoff_freq,
                                                              plot=plot, imz_flag=imz_flag)
    euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
    return noise_scale, isoline_snr, euclid, manhttan, cosine





#isoline_drift_butter(ecg_line,fs,imz_flag=True,filter_order=4,cutoff_freq=[2,200])
#isoline_drift_firwin(ecg_line,fs,imz_flag=True,filter_order=3000)
# like_2_staged_baseline_filter(ecg_line,fs,imz_flag=False,quality_factor=0.005,
#                               cutoff_freq=0.01,plot=True)
signal_filtering_functions.signal_FFT_filtration(signal_noised,fs,cutoff_freq=[1,200])