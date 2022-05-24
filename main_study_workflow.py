import numpy as np
import time

import Frequency_tools.Wavelet.CWT
import Geometry.Signal_geometry
import Plottermaan_lib
import signal_filtering_functions
from ECG_generation_func import ECG_generation
import Plottermaan_lib as pl
import noise_generation_functions as noise
from signal_filtering_functions import signal_bandpass_filtration,signal_highpass_filtration,\
    irnotch_filter,signal_FFT_filtration,signal_lopass_filtration
from scipy import signal
import matplotlib.pyplot as plt
import neurokit2 as nk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import distance

#!!!!!!!!!!!!!!111
#Нужно сравнивать не сигналы а кардиоцыклы, нужно добавить еще параметр схожести чтобы он отображал все изьяны может амплитуда разностной линии

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
    show_steps = True
)

from scipy import signal


def firwin_lowpass(cutoff, fs, order=3):
    if order % 2 == 0:
        order = order + 1
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b = signal.firwin(order, normal_cutoff)
    a = 1
    return b, a

def butter_lowpass(cutoff, fs, order=5,):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

b,a = butter_lowpass(15,fs,order=5)
filtered_signal = signal.filtfilt(b,a,ecg_line)
_, rpeaks = nk.ecg_peaks(filtered_signal, sampling_rate=fs)
signal_dwt, waves_dwt = nk.ecg_delineate(filtered_signal, rpeaks, sampling_rate=fs, method="dwt", show=False, show_type='all')
waves_dwt['ECG_R_Peaks'] = list(rpeaks['ECG_R_Peaks'])
ecg_points = waves_dwt
#Plottermaan_lib.masplot(ecg_line,fs=fs,markers=ecg_points,dotstyle='large')
Plottermaan_lib.masplot(ecg_line,filtered_signal,fs=fs,markers=ecg_points,dotstyle='large')
input('ss')
#Frequency_tools.Wavelet.CWT.wavelet_CWT(ecg_line,fs,scales_ramge=[1,8],time_range=[2.900,3],plott=True,block=True)
#Frequency_tools.Wavelet.CWT.morlet_CWT(ecg_line,fs,w=5,time_range=[1.43,1.5],freq_range=[0,200],plott=True,quality_factor=10)
#input('ss')
Geometry.Signal_geometry.MA_inflection_detector(ecg_line,fs,time_range=[2.558,2.763],plot=True,average_scale_coef=10)
input('ss')
'''
Noising ECG signal
'''
_,isoline_drift_corline,signal_noised = noise.signal_sine_noiser_generator(ecg_line,fs,0.25,scale=0.8,shift=0)
_,hz50_corline,signal_noised = noise.signal_sine_noiser_generator(ecg_line,fs,50,scale=0.05,shift=0)
_,white_noise_corline,signal_noised = noise.signal_white_noise_generator(signal_noised,scale=0.02,shift=0)
_,miosignal_corline,signal_noised = noise.signal_mio_noise_generator(signal_noised,fs=fs,scale=0.15,shift=0)

# pl.masplot(ecg_line,isoline_drift_corline,hz50_corline,white_noise_corline,miosignal_corline,signal_noised,
#            line_names=['Clean_ecg', 'Isoline drift noise','50Hz noise', 'White noise','Mio noise', 'Noised signal'],
#            fs=fs, dotstyle='large',
#            fig_title='Generation of noised signal',
#            line_width=3, y_axis_names=['Amplitude mV', 'Amplitude mV','Amplitude mV', 'Amplitude mV', 'Amplitude mV', 'Amplitude mV'],
#            x_axis_names=['Time, sec', 'Time, sec', 'Time, sec', 'Time, sec', 'Time, sec', 'Time, sec']
#            )
"""
ISOLINE DRIFT
"""
if False:
    print('#--------------------------------ISOLINE DRIFT--------------------------------#')
    def isoline_drift_butter(input_ecg,fs,
                             noise_freq = 0.5,noise_scale = 0.8,
                             filter_order = 4, cutoff_freq = [1,100],plot = False,
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
                             filter_order = 2000, cutoff_freq = [1,100],plot = False,imz_flag = False):
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
                             quality_factor = 0.005, cutoff_freq = 0.01,plot = False,imz_flag = False):
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

    def FFT_baseline_filter(input_ecg,fs,
                             noise_freq = 0.5,noise_scale = 0.8, cutoff_freq = [1,100],plot = False):
        ecg_line = input_ecg
        # basline drift
        isoline_snr, isoline_drift_corline, signal_noised = noise.signal_sine_noiser_generator(ecg_line, fs, noise_freq,
                                                                                               scale=noise_scale, shift=0)
        # Filtration - bandpass butter
        print('Firwin_bandpass')

        isoline_butter_bandpass_filt = signal_FFT_filtration(signal_noised,
                                                                  fs,cutoff_freq=cutoff_freq,
                                                                  plot=plot)
        euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
        return noise_scale, isoline_snr, euclid, manhttan, cosine


    #isoline_drift_butter(ecg_line,fs,imz_flag=True,plot=True)
    #isoline_drift_firwin(ecg_line,fs,imz_flag=True,plot=True)
    #like_2_staged_baseline_filter(ecg_line,fs,imz_flag=False,quality_factor=0.005,
    #                              cutoff_freq=0.01,plot=True)
    #FFT_baseline_filter(ecg_line,fs,cutoff_freq=[1,100],plot=True)

    def count_time_isoline_drift(iterations = 100):
        start_time = time.time()
        for i in range(iterations):
            print(str(i)+'/'+str(iterations))
            signal_bandpass_filtration(signal_noised,
                                       fs, order=4,
                                       cutoff_freq=[1,100],
                                       plot=False, filter='butter', imz_flag=False)
        butter_time = (time.time() - start_time)
        start_time = time.time()
        for i in range(iterations):
            print(str(i) + '/' + str(iterations))
            signal_bandpass_filtration(signal_noised,
                                       fs, order=2000,
                                       cutoff_freq=[1,100],
                                       plot=False, filter='firwin', imz_flag=False)
        firwin_time = (time.time() - start_time)
        start_time = time.time()
        for i in range(iterations):
            print(str(i) + '/' + str(iterations))
            isoline_butter_bandpass_filt = irnotch_filter(signal_noised,
                                                          fs, quality_factor=0.005,
                                                          cutoff_freq=0.01,
                                                          plot=False, imz_flag=False)
            signal.detrend(isoline_butter_bandpass_filt)
        irnotch_time = (time.time() - start_time)
        start_time = time.time()
        for i in range(iterations):
            print(str(i) + '/' + str(iterations))
            soline_butter_bandpass_filt = signal_FFT_filtration(signal_noised,
                                                                fs, cutoff_freq=[1, 100],
                                                                plot=False)
        fft_time = (time.time() - start_time)
        print("butter_time --- %s seconds ---" % (butter_time))
        print("firwin_time --- %s seconds ---" % (firwin_time))
        print("Irnotch_time --- %s seconds ---" % (irnotch_time))
        print("FFT_time --- %s seconds ---" % (fft_time))

        return butter_time,firwin_time,irnotch_time,fft_time

    def isoline_drift_bars_plot():
        _, _, butter_euclid, butter_manhttan, butter_cosine = isoline_drift_butter(ecg_line,fs)
        _,_,firwin_euclid, firwin_manhttan,firwin_cosine = isoline_drift_firwin(ecg_line,fs)
        _,_,staged_2_euclid, staged_2_manhttan,staged_2_cosine = like_2_staged_baseline_filter(ecg_line,fs)
        _,_,fft_euclid, fft_manhttan,fft_cosine = FFT_baseline_filter(ecg_line,fs)
        butter_time, firwin_time, irnotch_time, fft_time = count_time_isoline_drift()


        fig = make_subplots(rows=1, cols=5)

        categories = ['Euclid distance']
        row = 1
        col = 1
        fig.add_trace(go.Bar(name='Butter', x=categories, y=[butter_euclid],showlegend = False,marker_color='#0FB3FA'),row=row,col=col)
        fig.add_trace(go.Bar(name='Firwin', x=categories, y=[firwin_euclid],showlegend = False,marker_color='#43FA1B'), row=row, col=col)
        fig.add_trace(go.Bar(name='Iirnotch', x=categories, y=[staged_2_euclid],showlegend = False,marker_color='#FAC11B'), row = row, col = col)
        fig.add_trace(go.Bar(name='FFT filter', x=categories, y=[fft_euclid],showlegend = False,marker_color='#FA10F7'), row = row, col = col)

        categories = ['Manhattan distance']
        row = 1
        col = 2
        fig.add_trace(go.Bar(name='Butter', x=categories, y=[butter_manhttan],showlegend = False,marker_color='#0FB3FA'), row=row, col=col)
        fig.add_trace(go.Bar(name='Firwin', x=categories, y=[firwin_manhttan],showlegend = False,marker_color='#43FA1B'), row=row, col=col)
        fig.add_trace(go.Bar(name='Iirnotch', x=categories, y=[staged_2_manhttan],showlegend = False,marker_color='#FAC11B'), row=row, col=col)
        fig.add_trace(go.Bar(name='FFT filter', x=categories, y=[fft_manhttan],showlegend = False,marker_color='#FA10F7'), row=row, col=col)

        categories = ['Cosine similarity']
        row = 1
        col = 3
        fig.add_trace(go.Bar(name='Butter', x=categories, y=[butter_cosine],showlegend = False,marker_color='#0FB3FA'), row=row, col=col)
        fig.add_trace(go.Bar(name='Firwin', x=categories, y=[firwin_cosine],showlegend = False,marker_color='#43FA1B'), row=row, col=col)
        fig.add_trace(go.Bar(name='Iirnotch', x=categories, y=[staged_2_cosine],showlegend = False,marker_color='#FAC11B'), row=row, col=col)
        fig.add_trace(go.Bar(name='FFT filter', x=categories, y=[fft_cosine],showlegend = False,marker_color='#FA10F7'), row=row, col=col)

        categories = ['Manhattan distance']
        row = 1
        col = 4
        fig.add_trace(go.Bar(name='Butter', x=categories, y=[butter_manhttan],showlegend = False,marker_color='#0FB3FA'), row=row, col=col)
        fig.add_trace(go.Bar(name='Firwin', x=categories, y=[firwin_manhttan],showlegend = False,marker_color='#43FA1B'), row=row, col=col)
        fig.add_trace(go.Bar(name='Iirnotch', x=categories, y=[staged_2_manhttan],showlegend = False,marker_color='#FAC11B'), row=row, col=col)
        fig.add_trace(go.Bar(name='FFT filter', x=categories, y=[fft_manhttan],showlegend = False,marker_color='#FA10F7'), row=row, col=col)


        categories = ['Processing time']
        row = 1
        col = 5
        fig.add_trace(go.Bar(name='Butter', x=categories, y=[butter_time],marker_color='#0FB3FA'), row=row, col=col)
        fig.add_trace(go.Bar(name='Firwin', x=categories, y=[firwin_time],marker_color='#43FA1B'), row=row, col=col)
        fig.add_trace(go.Bar(name='Iirnotch', x=categories, y=[irnotch_time],marker_color='#FAC11B'), row=row, col=col)
        fig.add_trace(go.Bar(name='FFT filter', x=categories, y=[fft_time],marker_color='#FA10F7'), row=row, col=col)
        fig.update_layout(barmode='group')

        fig.update_layout(barmode='group')
        fig.show()


    #isoline_drift_bars_plot()
    def baseline_filters_test():
        low_noise_scale = 0
        high_noise_scale = 5
        step = 0.1
        scales_mas = np.arange(low_noise_scale,high_noise_scale,step)
        #Butter
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine = isoline_drift_butter(ecg_line, fs, imz_flag=False, filter_order=4,
                                                                  cutoff_freq=[1, 100], plot=False,noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        butter_snr_mass = np.array(snr_mass)
        butter_euclid_mass = np.array(euclid_mass)
        butter_manhtan_mass = np.array(manhttan_mass)
        butter_cosine_mass = np.array(cosine_mass)

        # firwin
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine = isoline_drift_firwin(ecg_line,fs,imz_flag=False,filter_order=2000,cutoff_freq=[1,100],
                                                                    plot=False,noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        firwin_snr_mass = np.array(snr_mass)
        firwin_euclid_mass = np.array(euclid_mass)
        firwin_manhtan_mass = np.array(manhttan_mass)
        firwin_cosine_mass = np.array(cosine_mass)

        # 2 staged
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine =  like_2_staged_baseline_filter(ecg_line,fs,imz_flag=False,
                                                                              quality_factor=0.005,cutoff_freq=0.01,
                                                                              plot=False,noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        staged2_snr_mass = np.array(snr_mass)
        staged2_euclid_mass = np.array(euclid_mass)
        staged2_manhtan_mass = np.array(manhttan_mass)
        staged2_cosine_mass = np.array(cosine_mass)

        # fft
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine =  FFT_baseline_filter(ecg_line,fs,cutoff_freq=[1,100],plot=False,noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        FFT_snr_mass = np.array(snr_mass)
        FFT_euclid_mass = np.array(euclid_mass)
        FFT_manhtan_mass = np.array(manhttan_mass)
        FFT_cosine_mass = np.array(cosine_mass)

        plt.figure()
        ax1 = plt.subplot(2,2,1)
        plt.plot(butter_snr_mass,butter_euclid_mass)
        plt.plot(butter_snr_mass,firwin_euclid_mass)
        plt.plot(butter_snr_mass,staged2_euclid_mass)
        plt.plot(butter_snr_mass,FFT_euclid_mass)
        plt.legend(["Butter", "Firwin","Iirnotch","FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Euclidian distance')

        ax1 = plt.subplot(2, 2, 2)
        plt.plot(butter_snr_mass, butter_manhtan_mass)
        plt.plot(butter_snr_mass, firwin_manhtan_mass)
        plt.plot(butter_snr_mass, staged2_manhtan_mass)
        plt.plot(butter_snr_mass, FFT_manhtan_mass)
        plt.legend(["Butter", "Firwin", "Iirnotch", "FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Manhtan distance')

        ax1 = plt.subplot(2, 2, (3,4))
        plt.plot(butter_snr_mass, butter_cosine_mass)
        plt.plot(butter_snr_mass, firwin_cosine_mass)
        plt.plot(butter_snr_mass, staged2_cosine_mass)
        plt.plot(butter_snr_mass, FFT_cosine_mass)
        plt.legend(["Butter", "Firwin", "Iirnotch", "FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Cosine similarity')
        plt.show()

    baseline_filters_test()

"""
SINE 50 HZ
"""
if False:
    print('#--------------------------------SINE 50 HZ--------------------------------#')
    def hz50_butter(input_ecg,fs,
                             noise_freq = 50,noise_scale = 0.05,
                             filter_order = 3, cutoff_freq = 40,plot = False,
                             imz_flag =False):
        ecg_line = input_ecg
        snr, hz50_corline, signal_noised = noise.signal_sine_noiser_generator(ecg_line, fs, noise_freq, scale=noise_scale, shift=0)

        #Filtration - bandpass butter
        print('Butter_bandpass')
        isoline_butter_bandpass_filt = signal_lopass_filtration(signal_noised,
                                                                  fs,order=filter_order,
                                                                  cutoff_freq=cutoff_freq,
                                                                  plot=plot,filter='butter',imz_flag=imz_flag)
        euclid,manhttan,cosine = ecg_signals_comparsion(ecg_line,isoline_butter_bandpass_filt,fs,plot=plot)
        return noise_scale,snr,euclid,manhttan,cosine

    def hz_50_irnotch_filter(input_ecg,fs,
                             noise_freq = 50,noise_scale = 0.05,
                             quality_factor = 30, cutoff_freq = 50,plot = False,imz_flag = False):
        ecg_line = input_ecg
        # basline drift
        snr, hz50_corline, signal_noised = noise.signal_sine_noiser_generator(ecg_line, fs, noise_freq, scale=noise_scale, shift=0)
        # Filtration - bandpass butter
        print('Firwin_bandpass')

        isoline_butter_bandpass_filt = irnotch_filter(signal_noised,
                                                                  fs, quality_factor=quality_factor,
                                                                  cutoff_freq=cutoff_freq,
                                                                  plot=plot, imz_flag=imz_flag,detrend=False)
        euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
        return noise_scale, snr, euclid, manhttan, cosine

    def FFT_hz_50_filter(input_ecg,fs,
                             noise_freq = 50,noise_scale = 0.05, cutoff_freq = [49.8,50.2],plot = False):
        ecg_line = input_ecg
        # basline drift
        snr, hz50_corline, signal_noised = noise.signal_sine_noiser_generator(ecg_line, fs, noise_freq, scale=noise_scale, shift=0)
        # Filtration - bandpass butter
        print('Firwin_bandpass')
        isoline_butter_bandpass_filt = signal_FFT_filtration(signal_noised,
                                                                  fs,cutoff_freq=cutoff_freq,bandstop=True,
                                                                  plot=plot)
        euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
        return noise_scale, snr, euclid, manhttan, cosine

    #hz50_butter(ecg_line,fs,plot=True,imz_flag=True)
    #hz_50_irnotch_filter(ecg_line,fs,imz_flag=True,plot=True)
    #FFT_hz_50_filter(ecg_line,fs,plot=True)

    def baseline_filters_test():
        low_noise_scale = 0
        high_noise_scale = 0.8
        step = 0.1
        scales_mas = np.arange(low_noise_scale,high_noise_scale,step)
        #Butter
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine = hz50_butter(ecg_line, fs, imz_flag=False, filter_order=4,
                                                                  noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        butter_snr_mass = np.array(snr_mass)
        butter_euclid_mass = np.array(euclid_mass)
        butter_manhtan_mass = np.array(manhttan_mass)
        butter_cosine_mass = np.array(cosine_mass)

        # 2 staged
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine =  hz_50_irnotch_filter(ecg_line,fs,imz_flag=False,
                                                                              plot=False,noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        staged2_snr_mass = np.array(snr_mass)
        staged2_euclid_mass = np.array(euclid_mass)
        staged2_manhtan_mass = np.array(manhttan_mass)
        staged2_cosine_mass = np.array(cosine_mass)

        # fft
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine =  FFT_hz_50_filter(ecg_line,fs,plot=False,noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        FFT_snr_mass = np.array(snr_mass)
        FFT_euclid_mass = np.array(euclid_mass)
        FFT_manhtan_mass = np.array(manhttan_mass)
        FFT_cosine_mass = np.array(cosine_mass)

        plt.figure()
        ax1 = plt.subplot(2,2,1)
        plt.plot(butter_snr_mass,butter_euclid_mass)
        plt.plot(butter_snr_mass,staged2_euclid_mass)
        plt.plot(butter_snr_mass,FFT_euclid_mass)
        plt.legend(["Butter","Iirnotch","FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Euclidian distance')

        ax1 = plt.subplot(2, 2, 2)
        plt.plot(butter_snr_mass, butter_manhtan_mass)
        plt.plot(butter_snr_mass, staged2_manhtan_mass)
        plt.plot(butter_snr_mass, FFT_manhtan_mass)
        plt.legend(["Butter", "Iirnotch", "FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Manhtan distance')

        ax1 = plt.subplot(2, 2, (3,4))
        plt.plot(butter_snr_mass, butter_cosine_mass)
        plt.plot(butter_snr_mass, staged2_cosine_mass)
        plt.plot(butter_snr_mass, FFT_cosine_mass)
        plt.legend(["Butter","Iirnotch", "FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Cosine similarity')
        plt.show()

    baseline_filters_test()

"""
White noise
"""

if False:
    def White_noise_butter(input_ecg, fs, noise_scale=0.02,
                             filter_order=4, cutoff_freq=[1, 100], plot=False,
                             imz_flag=False):
        ecg_line = input_ecg
        isoline_snr, white_noise_corline, signal_noised = noise.signal_white_noise_generator(ecg_line, scale=noise_scale, shift=0)
        # Filtration - bandpass butter
        print('Butter_bandpass')
        isoline_butter_bandpass_filt = signal_bandpass_filtration(signal_noised,
                                                                  fs, order=filter_order,
                                                                  cutoff_freq=cutoff_freq,
                                                                  plot=plot, filter='butter', imz_flag=imz_flag)
        euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
        return noise_scale, isoline_snr, euclid, manhttan, cosine


    def White_noise_firwin(input_ecg, fs, noise_scale=0.02,
                             filter_order=2000, cutoff_freq=[1, 100], plot=False, imz_flag=False):
        ecg_line = input_ecg
        isoline_snr, white_noise_corline, signal_noised = noise.signal_white_noise_generator(ecg_line, scale=noise_scale, shift=0)
        # Filtration - bandpass butter
        print('Firwin_bandpass')

        isoline_butter_bandpass_filt = signal_bandpass_filtration(signal_noised,
                                                                  fs, order=filter_order,
                                                                  cutoff_freq=cutoff_freq,
                                                                  plot=plot, filter='firwin', imz_flag=imz_flag)
        euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
        return noise_scale, isoline_snr, euclid, manhttan, cosine


    def FFT_White_noise_filter(input_ecg, fs, noise_scale=0.02, cutoff_freq=[1, 100], plot=False):
        ecg_line = input_ecg
        isoline_snr, white_noise_corline, signal_noised = noise.signal_white_noise_generator(ecg_line, scale=noise_scale, shift=0)
        # Filtration - bandpass butter
        print('Firwin_bandpass')

        isoline_butter_bandpass_filt = signal_FFT_filtration(signal_noised,
                                                             fs, cutoff_freq=cutoff_freq, #treshold=10,
                                                             plot=plot)
        euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
        return noise_scale, isoline_snr, euclid, manhttan, cosine

    #White_noise_butter(ecg_line,fs,plot=True,imz_flag=True)
    # White_noise_firwin(ecg_line,fs,plot=True,imz_flag=True)
    # FFT_White_noise_filter(ecg_line,fs,plot=True)

    def baseline_filters_test():
        low_noise_scale = 0
        high_noise_scale = 0.1
        step = 0.005
        scales_mas = np.arange(low_noise_scale,high_noise_scale,step)
        #Butter
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine = White_noise_butter(ecg_line, fs, imz_flag=False, plot=False,noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        butter_snr_mass = np.array(snr_mass)
        butter_euclid_mass = np.array(euclid_mass)
        butter_manhtan_mass = np.array(manhttan_mass)
        butter_cosine_mass = np.array(cosine_mass)

        # firwin
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine = White_noise_firwin(ecg_line,fs,imz_flag=False,
                                                                    plot=False,noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        firwin_snr_mass = np.array(snr_mass)
        firwin_euclid_mass = np.array(euclid_mass)
        firwin_manhtan_mass = np.array(manhttan_mass)
        firwin_cosine_mass = np.array(cosine_mass)

        # fft
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine = FFT_White_noise_filter(ecg_line,fs,plot=False,noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        FFT_snr_mass = np.array(snr_mass)
        FFT_euclid_mass = np.array(euclid_mass)
        FFT_manhtan_mass = np.array(manhttan_mass)
        FFT_cosine_mass = np.array(cosine_mass)

        plt.figure()
        ax1 = plt.subplot(2,2,1)
        plt.plot(butter_snr_mass,butter_euclid_mass)
        plt.plot(butter_snr_mass,firwin_euclid_mass)
        plt.plot(butter_snr_mass,FFT_euclid_mass)
        plt.legend(["Butter", "Firwin","FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Euclidian distance')

        ax1 = plt.subplot(2, 2, 2)
        plt.plot(butter_snr_mass, butter_manhtan_mass)
        plt.plot(butter_snr_mass, firwin_manhtan_mass)
        plt.plot(butter_snr_mass, FFT_manhtan_mass)
        plt.legend(["Butter", "Firwin", "FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Manhtan distance')

        ax1 = plt.subplot(2, 2, (3,4))
        plt.plot(butter_snr_mass, butter_cosine_mass)
        plt.plot(butter_snr_mass, firwin_cosine_mass)
        plt.plot(butter_snr_mass, FFT_cosine_mass)
        plt.legend(["Butter", "Firwin", "FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Cosine similarity')
        plt.show()

    baseline_filters_test()

"""
Mio noise
"""

if False:
    def MIO_noise_butter(input_ecg, fs, noise_scale=0.15,
                           filter_order=4, cutoff_freq=[1, 100], plot=False,
                           imz_flag=False):
        ecg_line = input_ecg

        isoline_snr, miosignal_corline, signal_noised = noise.signal_mio_noise_generator(ecg_line, fs=fs,scale=noise_scale, shift=0)

        # Filtration - bandpass butter
        print('Butter_bandpass')
        isoline_butter_bandpass_filt = signal_bandpass_filtration(signal_noised,
                                                                  fs, order=filter_order,
                                                                  cutoff_freq=cutoff_freq,
                                                                  plot=plot, filter='butter', imz_flag=imz_flag)
        euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
        return noise_scale, isoline_snr, euclid, manhttan, cosine


    def MIO_noise_firwin(input_ecg, fs, noise_scale=0.15,
                           filter_order=2000, cutoff_freq=[1, 100], plot=False, imz_flag=False):
        ecg_line = input_ecg
        isoline_snr, miosignal_corline, signal_noised = noise.signal_mio_noise_generator(ecg_line, fs=fs,scale=noise_scale, shift=0)
        # Filtration - bandpass butter
        print('Firwin_bandpass')

        isoline_butter_bandpass_filt = signal_bandpass_filtration(signal_noised,
                                                                  fs, order=filter_order,
                                                                  cutoff_freq=cutoff_freq,
                                                                  plot=plot, filter='firwin', imz_flag=imz_flag)
        euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
        return noise_scale, isoline_snr, euclid, manhttan, cosine


    def FFT_MIO_noise_filter(input_ecg, fs, noise_scale=0.15, cutoff_freq=[1, 100], plot=False):
        ecg_line = input_ecg
        isoline_snr, miosignal_corline, signal_noised = noise.signal_mio_noise_generator(ecg_line, fs=fs,scale=noise_scale, shift=0)
        # Filtration - bandpass butter
        print('Firwin_bandpass')

        isoline_butter_bandpass_filt = signal_FFT_filtration(signal_noised,
                                                             fs, cutoff_freq=cutoff_freq,  #treshold=10,
                                                             plot=plot)
        euclid, manhttan, cosine = ecg_signals_comparsion(ecg_line, isoline_butter_bandpass_filt, fs, plot=plot)
        return noise_scale, isoline_snr, euclid, manhttan, cosine


    #MIO_noise_butter(ecg_line,fs,plot=True,imz_flag=True)
    # MIO_noise_firwin(ecg_line,fs,plot=True,imz_flag=True)
    #FFT_MIO_noise_filter(ecg_line,fs,plot=True)

    def baseline_filters_test():
        low_noise_scale = 0
        high_noise_scale = 0.2
        step = 0.005
        scales_mas = np.arange(low_noise_scale, high_noise_scale, step)
        # Butter
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine = MIO_noise_butter(ecg_line, fs, imz_flag=False, plot=False,
                                                                  noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        butter_snr_mass = np.array(snr_mass)
        butter_euclid_mass = np.array(euclid_mass)
        butter_manhtan_mass = np.array(manhttan_mass)
        butter_cosine_mass = np.array(cosine_mass)

        # firwin
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine = MIO_noise_firwin(ecg_line, fs, imz_flag=False,
                                                                  plot=False, noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        firwin_snr_mass = np.array(snr_mass)
        firwin_euclid_mass = np.array(euclid_mass)
        firwin_manhtan_mass = np.array(manhttan_mass)
        firwin_cosine_mass = np.array(cosine_mass)

        # fft
        snr_mass = []
        euclid_mass = []
        manhttan_mass = []
        cosine_mass = []
        for scale in scales_mas:
            print(scale)
            _, snr, euclid, manhttan, cosine = FFT_MIO_noise_filter(ecg_line, fs, plot=False, noise_scale=scale)
            snr_mass.append(snr)
            euclid_mass.append(euclid)
            manhttan_mass.append(manhttan)
            cosine_mass.append(cosine)

        FFT_snr_mass = np.array(snr_mass)
        FFT_euclid_mass = np.array(euclid_mass)
        FFT_manhtan_mass = np.array(manhttan_mass)
        FFT_cosine_mass = np.array(cosine_mass)

        plt.figure()
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(butter_snr_mass, butter_euclid_mass)
        plt.plot(butter_snr_mass, firwin_euclid_mass)
        plt.plot(butter_snr_mass, FFT_euclid_mass)
        plt.legend(["Butter", "Firwin", "FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Euclidian distance')

        ax1 = plt.subplot(2, 2, 2)
        plt.plot(butter_snr_mass, butter_manhtan_mass)
        plt.plot(butter_snr_mass, firwin_manhtan_mass)
        plt.plot(butter_snr_mass, FFT_manhtan_mass)
        plt.legend(["Butter", "Firwin", "FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Manhtan distance')

        ax1 = plt.subplot(2, 2, (3, 4))
        plt.plot(butter_snr_mass, butter_cosine_mass)
        plt.plot(butter_snr_mass, firwin_cosine_mass)
        plt.plot(butter_snr_mass, FFT_cosine_mass)
        plt.legend(["Butter", "Firwin", "FFT filter"])
        plt.xlabel('SNR dB')
        plt.ylabel('Param_value')
        ax1.title.set_text('Cosine similarity')
        plt.show()


    baseline_filters_test()
