

'''
Polarization analysis.
April 2022
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

    return arr_phi, beam #power

# angles = [200,210,215,220,225,230,235,240,245,250,255,260,270,275,280,285,290,295,300,310,320,330,340,350,360,370]
# angles = [200,210,215,220,225]+list(np.array([230,235,240,245,250,255,260,270,275,280,285,290,295,300,310,320,330,340,350,360,370])/2)
angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]
# ang = [45,55, 65, 75, 225, 235, 245, 255, 265]
# angles = angles + ang
angles = list(np.arange(0, 360, 10))
ang = [55, 65, 75, 85, 235, 245, 255, 265]
angles = angles + ang
beams = []

###
F_test = [145, 145,150,155,160,165,170] # GHz
beams_f = np.zeros((len(F_test), len(angles)))
for i, freq in enumerate(F_test):
    beambeam = []
    for j, ang in enumerate(angles):
        phi, beam = import_pol_data(f"../Data/pol_{freq}GHz_"+str(ang)+"deg_21-4-2022.txt",5)
        beambeam.append(beam)
    beams_f[i] = np.array(beambeam)
gridsize = int(np.sqrt((len(F_test)))+0.5)

fig, ax = plt.subplots(gridsize, gridsize, figsize=(15, 15))
for ib, b in enumerate(beams_f):
    ax[ib//gridsize, int(np.mod(ib, gridsize))].plot(angles,np.array(b)/np.max(np.array(b)),'.',color = 'r')
    ax[ib//gridsize, int(np.mod(ib, gridsize))].set_title(f"frequency : {F_test[ib]} GHz")
    ax[ib//gridsize, int(np.mod(ib, gridsize))].set_xlabel("Grid tilt [deg.]")
    ax[ib//gridsize, int(np.mod(ib, gridsize))].set_yscale("log")
plt.show()
###
#
# for ang in angles:
#
#     phi,beam = import_pol_data("../Data/pol_150GHz_"+str(ang)+"deg_21-4-2022.txt",5)
#     beams.append(beam)
#
# plt.plot(angles,np.array(beams)/np.max(np.array(beams)),'.',color = 'r')
# plt.title("Polarization Response of Holography Receiver")
# plt.xlabel("Grid tilt [deg.]")
# plt.yscale("log")
# plt.show()
