import warnings
warnings.filterwarnings('ignore')

import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import Plottermaan_lib as pl
from scipy import signal
import signal_filtering_functions as filt
import noise_generation_functions as noise

# #initial
# duration = 10
# hr = 80
# fs = 1000
#
# P_wave_splitting_flag = True
# R_wave_splitting_flag = True
# delta_wave_flag = True
# f_wave_flag = True
# extrasystole = True
# extrasystoly_n_central_points = 3
# show_steps = True
def ECG_generation(
    duration = 10,
    hr = 80,
    fs = 1000,
    P_wave_splitting_flag = True,
    R_wave_splitting_flag = True,
    delta_wave_flag = True,
    f_wave_flag = True,
    extrasystole = True,
    extrasystoly_n_central_points = 3,
    show_steps = True):
    #----------------------------------Generate ECG------------------------------------------

    def peak_splitting(ecg,fs,peaks_possition_list,surrounding_points_ms = 0,
                       splitting_offset_ms = 0,corection_line_Fiilter_order = 2,corection_line_Fiilter_feq = 100,scale_cof = 1):
        corection_line = np.zeros(np.shape(ecg)[0])
        surrounding_p_points = int(np.ceil((surrounding_points_ms/1000)*fs))
        spliting_offset = int(np.ceil((splitting_offset_ms/1000)*fs))
        for p_peak_position in peaks_possition_list:
            low_range = p_peak_position-surrounding_p_points
            hi_range = p_peak_position+surrounding_p_points+spliting_offset
            low_range2 = p_peak_position-surrounding_p_points+spliting_offset
            if low_range>0 and low_range2>0 and hi_range < np.shape(ecg)[0]:
                initial_p_part = ecg[p_peak_position-surrounding_p_points:p_peak_position+surrounding_p_points]

                corection_line[p_peak_position-surrounding_p_points+spliting_offset:
                               p_peak_position+surrounding_p_points+spliting_offset] = initial_p_part
        #Filtering_coefs_setup
        nyq = 0.5 * fs
        corection_line_Fiilter_feq = corection_line_Fiilter_feq/nyq
        vawe_split_a,vawe_split_b = signal.butter(corection_line_Fiilter_order,
                                                  corection_line_Fiilter_feq, 'low')
        corection_line_filtered = signal.filtfilt(vawe_split_a,vawe_split_b,corection_line)
        return corection_line,corection_line_filtered*scale_cof

    def f_vawe_generation(ecg,fs,peaks_possition_list,scale_cof = 1,offset = 0,corection_line_Fiilter_feq = 10,corection_line_Fiilter_order = 2):
        corection_line = np.zeros(np.shape(ecg)[0])
        for peak in peaks_possition_list:
            f = 15
            t = 0.2
            samples = np.arange(t * fs)
            signall = np.sin(samples*2*3.14*f)
            samples_len = np.shape(samples)[0]
            if (peak+samples_len)<np.shape(ecg)[0]:
                signal_abs = np.abs(signall)
                signal_abs = signal_abs+offset
                corection_line[peak:peak+samples_len] = signal_abs
        nyq = 0.5 * fs
        corection_line_Fiilter_feq = corection_line_Fiilter_feq / nyq
        vawe_split_a, vawe_split_b = signal.butter(corection_line_Fiilter_order,
                                                   corection_line_Fiilter_feq, 'low')
        corection_line = signal.filtfilt(vawe_split_a, vawe_split_b, corection_line)
        return corection_line*scale_cof

    #initial ecg
    ecg = nk.ecg_simulate(duration=duration,sampling_rate=fs,noise=0, heart_rate=hr,method= 'ecgsyn')
    b_drift, a_drift = signal.iirnotch(0.01,  Q = 0.005, fs = fs)# baseline_drifft_filt
    ecg = signal.detrend(signal.filtfilt(b_drift,a_drift,ecg))# deetrend
    t = np.arange(0,(np.shape(ecg)[0]/fs),(1/fs))

    #peaks detector
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=fs,)
    signal_dwt, waves_dwt = nk.ecg_delineate(ecg, rpeaks, sampling_rate=fs, method="dwt", show=False, show_type='all')
    waves_dwt['ECG_R_Peaks'] = list(rpeaks['ECG_R_Peaks'])
    ecg_points = waves_dwt


    #creation peak splitting function


    debug_cor1,p_vawe_split_corection_line = peak_splitting(ecg,fs,
            waves_dwt['ECG_P_Peaks'],
            surrounding_points_ms= 5,
            splitting_offset_ms=50,
            corection_line_Fiilter_order=3,
            corection_line_Fiilter_feq=20,
            scale_cof=2.3)


    debug_cor2,R_vawe_split_corection_line = peak_splitting(ecg,fs,
            waves_dwt['ECG_R_Peaks'],
            surrounding_points_ms= 5,
            splitting_offset_ms=25,
            corection_line_Fiilter_order=1,
            corection_line_Fiilter_feq=50,
            scale_cof=1)

    debug_cor3,delta_vawe_split_corection_line = peak_splitting(ecg,fs,
            waves_dwt['ECG_R_Peaks'],
            surrounding_points_ms= 7,
            splitting_offset_ms=-35,
            corection_line_Fiilter_order=1,
            corection_line_Fiilter_feq=30,
            scale_cof=0.4)

    f_vawe = f_vawe_generation(ecg,fs,
                               waves_dwt['ECG_T_Offsets'],
                               scale_cof=0.2,offset=-0.35,
                               corection_line_Fiilter_feq = 30,
                               corection_line_Fiilter_order = 10)


    point_side_num = int(np.floor(extrasystoly_n_central_points/2))
    central_point_num = int(np.ceil(np.shape(waves_dwt['ECG_R_Peaks'])[0]/2))
    points_num_list = list(range(central_point_num-point_side_num,central_point_num+point_side_num+1,1))
    extrasystole_position_mass = np.array(waves_dwt['ECG_R_Peaks'])[[points_num_list]]
    _,extrasystole_vawe = peak_splitting(ecg,fs,
            extrasystole_position_mass,
            surrounding_points_ms= 15,
            splitting_offset_ms=0,
            corection_line_Fiilter_order=1,
            corection_line_Fiilter_feq=30,
            scale_cof=0.8)


    resulting_ecg = ecg
    if extrasystole:
        resulting_ecg = resulting_ecg+extrasystole_vawe

    if P_wave_splitting_flag:
        resulting_ecg = resulting_ecg+p_vawe_split_corection_line

    if R_wave_splitting_flag:
        resulting_ecg = resulting_ecg+R_vawe_split_corection_line

    if delta_wave_flag:
        resulting_ecg = resulting_ecg+delta_vawe_split_corection_line

    if f_wave_flag:
        resulting_ecg = resulting_ecg+f_vawe

    corection_line = extrasystole_vawe+p_vawe_split_corection_line+R_vawe_split_corection_line \
    +delta_vawe_split_corection_line+f_vawe

    if show_steps:
        pl.masplot(ecg,debug_cor1,p_vawe_split_corection_line,ecg+p_vawe_split_corection_line,
                   line_names=['Clean_ecg','Targert part shifted','Targert part filtered','Resulting ECG'],
                   markers_names=False,
                   markers=ecg_points,fs=fs,dotstyle='large',
                   fig_title='Generation of P-peak splitting',
                   line_width=3,y_axis_names=['Amplitude mV','Amplitude mV','Amplitude mV','Amplitude mV'],
                   x_axis_names=['Time, sec','Time, sec','Time, sec','Time, sec'])
        pl.masplot(ecg,corection_line,resulting_ecg,markers=ecg_points, fs=fs,
                   dotstyle='large',line_names=['Clean_ecg','Correction line', 'Resulting ECG'],
                   markers_names=False, fig_title='Generation of patalogy ECG signal',
                   line_width=2.5, y_axis_names=['Amplitude mV','Amplitude mV','Amplitude mV'],
                   x_axis_names=['Time, sec','Time, sec','Time, sec']

                   )  # markers=ecg_points
    #Signal noising
    # signal_noised = noise.signal_sine_noiser_generator(resulting_ecg,1,scale=0.001,shift=0)
    # signal_noised = noise.signal_white_noise_generator(signal_noised,scale=0.02,shift=0)
    # signal_noised = noise.signal_mio_noise_generator(signal_noised,scale=0.15,shift=0)

    #signal_lopass_filtration(signal_noised,fs=fs,order=20,cutoff_freq=200,filter ='firwin')
    #signal_highpass_filtration(signal_noised,fs=fs,order=20,cutoff_freq=200,filter ='firwin')
    #signal_bandpass_filtration(signal_noised,fs=fs,order=60,cutoff_freq=[50,100],filter ='firwin')
    # signal_highpass_filtration(signal_noised,fs=fs,order=20,cutoff_freq=200)
    #signal_bandpass_filtration(signal_noised,fs=fs,order=10,cutoff_freq=[100,200])
    #pl.masplot(ecg,resulting_ecg,extrasystole_vawe,signal_noised,markers=ecg_points,fs=fs,dotstyle='large') #markers=ecg_points
    return resulting_ecg

