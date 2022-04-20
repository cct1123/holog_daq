

'''
Polarization analysis.
April 2022
'''

import numpy as np


def running_mean(arr, n_mean):
    """
    Compute the running mean of array.
    """
    return np.convolve(arr, np.ones((n_mean,)) / n_mean)[(n_mean - 1) :]


def import_pol_data(file, n_indiv):
    """
    Import .txt from ROACH2 and return beam data.
    """
    data = np.loadtxt(file, skiprows=1)

    l_mean = 1
    # n_indiv = 7 # change number to 5 if measurement taken before june 10 2021 because
    # format of data saved to file changed when we chose to also document
    # the motor's specified position at each step, therefore there are 7
    # variables recorded at the beginning of the file, whereas it used to be 5.

    line_size = np.size(data)
    nsamp = np.size(data, 0)


    i_aa_begin = int(n_indiv + (1 - 1) * (line_size - n_indiv) / 4)
    i_aa_end = int(n_indiv + (2 - 1) * (line_size - n_indiv) / 4) - 1
    i_bb_begin = int(n_indiv + (2 - 1) * (line_size - n_indiv) / 4)
    i_bb_end = int(n_indiv + (3 - 1) * (line_size - n_indiv) / 4) - 1
    i_ab_begin = int(n_indiv + (3 - 1) * (line_size - n_indiv) / 4)
    i_ab_end = int(n_indiv + (4 - 1) * (line_size - n_indiv) / 4) - 1
    i_phase_begin = int(n_indiv + (4 - 1) * (line_size - n_indiv) / 4)
    i_phase_end = int(n_indiv + (5 - 1) * (line_size - n_indiv) / 4) - 1

    i = int(0)

    arr_phi = data[3]

    arr_aa = np.array(running_mean(data[i_aa_begin:i_aa_end], l_mean))
    arr_bb = np.array(running_mean(data[i_bb_begin:i_bb_end], l_mean))
    arr_ab = np.array(running_mean(data[i_ab_begin:i_ab_end], l_mean))
    arr_phase = np.array(data[i_phase_begin:i_phase_end])
    n_channels = np.size(arr_ab)

    # make amplitude arrays, in case they need to be plotted.
    amp_cross = np.power(arr_ab[int(n_channels / 2)], 1)
    amp_var = np.power(
        np.divide(arr_ab[int(n_channels / 2)], arr_aa[int(n_channels / 2)]), 2
    )
    amp_aa = arr_aa[int(n_channels / 2)]
    amp_bb = arr_bb[int(n_channels / 2)]


    amp = amp_var
    amp = np.divide(amp, np.max(amp))

    es_arr = amp_cross
    ss_arr = amp_aa

    source = ss_arr / np.max(ss_arr)
    beam = es_arr ** 2 / source

    return arr_phi, np.sqrt(beam)

phi,beam = import_pol_data("../Data/pol_130GHz_170deg_20-4-2022.txt",5)
print(phi,beam)
