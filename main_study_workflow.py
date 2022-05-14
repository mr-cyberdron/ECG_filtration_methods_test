from ECG_generation_func import ECG_generation
import Plottermaan_lib as pl
import noise_generation_functions as noise
'''
Generate initial ECG
'''
ecg_line = ECG_generation(
    duration = 10,
    hr = 80,
    fs = 1000,
    P_wave_splitting_flag = True,
    R_wave_splitting_flag = True,
    delta_wave_flag = True,
    f_wave_flag = True,
    extrasystole = True,
    extrasystoly_n_central_points = 3,
    show_steps = True
)
pl.masplot(ecg_line)
'''
Noising ECG signal
'''
