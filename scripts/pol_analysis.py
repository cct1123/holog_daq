

'''
Polarization analysis.
April 2022
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import rand

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

def random_color():
    """
    Generate a random color.
    """
    r = np.random.rand()
    g = np.random.rand()
    b = np.random.rand()
    return (r, g, b)

from scipy.optimize import curve_fit
def sine_func(x, a, b, c, d):
    """
    Sine function.
    """
    return a * np.sin(b * x + c) + d + 1

def sine_fitting(x, y):
    """
    Fit a sine function to data.
    """
    # estimate parameters
    base = np.mean(y)
    amp = np.max(y) - base
    # phase = np.arctan2(y[0], x[0])
    phase = 0
    freq = 1.5 * np.pi / 180.0

    # fit
    popt, pcov = curve_fit(sine_func, x, y,  
                            p0=(amp, freq, phase, base), 
                            # bounds=([0, 0, -2*np.pi, 0], [10*amp, 10*freq, 2*np.pi, 10*base]), 
                            # method='trf'
                            )

    x_fit = np.linspace(x[0], x[-1], 500)
    y_fit = sine_func(x_fit, *popt)
    return x_fit, y_fit


# angles = [200,210,215,220,225,230,235,240,245,250,255,260,270,275,280,285,290,295,300,310,320,330,340,350,360,370]
# angles = [200,210,215,220,225]+list(np.array([230,235,240,245,250,255,260,270,275,280,285,290,295,300,310,320,330,340,350,360,370])/2)

if __name__ == "__main__":
    path = "D:/holog_daq"
    freq_list_low = [135, 140]  # GHz
    freq_list_high = [145,150,155,160,165,170] # GHz
    freq_array = np.array(freq_list_low + freq_list_high)

    angles_low = np.sort(list(np.arange(0, 360, 10)) +  [45,55, 65, 75, 225, 235, 245, 255, 265])# angles for freq 130,135,140
    angles_high = np.sort(list(np.arange(0, 360, 10)) + [55, 65, 75, 85, 235, 245, 255, 265]) # angles for freq 150,155,160, 165, 170

    angles_array = np.array([angles_low]*len(freq_list_low) + [angles_high]*len(freq_list_high))
    beams_array = np.zeros_like(angles_array)

    for i, freq in enumerate(freq_array):
        beambeam = []
        for j, ang in enumerate(angles_array[i]):
            phi, beam = import_pol_data(f"{path}/Data/pol_{freq}GHz_"+str(ang)+"deg_21-4-2022.txt",5)
            beambeam.append(beam)
        beams_array[i] = np.array(beambeam)

    crospratio_array = np.zeros_like(freq_array, dtype=np.float)
    for ib, b in enumerate(beams_array):
        y = np.array(b)
        crosp = np.min(y)
        cop = np.max(y)
        print(crosp, cop)
        crospratio_array[ib] = np.true_divide(crosp, cop)*100 # in %

    print(crospratio_array)

    ########### plot ###############-------------------------------------------------------------------------------------
    # randcolorlist = [random_color() for i in range(len(freq_array))]
    randcolorlist = [
                        (0.5165371765038422, 0.04289909921364421, 0.9803930929495331), 
                        (0.7723185593932307, 0.7090741470712764, 0.3028260568485742), 
                        (0.9042597124184352, 0.2556604579473787, 0.49938166422355335),
                        (0.4715998027915179, 0.8847302805014586, 0.7928790654653481),
                        (0.5454975838075998, 0.6166934877163249, 0.28118964055877993),
                        (0.853588623387843, 0.5261775660455685, 0.2403459113645653),
                        (0.14788194132340793, 0.43610896071169625, 0.7449996130477309),
                        (0.7242586132468174, 0.47353845257824156, 0.9306233188168667)
                    ]
    # show grid of plots
    gridsize = int(np.sqrt((len(freq_array)))+0.5)
    fig, ax = plt.subplots(gridsize, gridsize, figsize=(9, 4))
    fig.suptitle("Polarization Response of Holography Receiver", fontsize=16)
    for ib, b in enumerate(beams_array):
        x = angles_array[ib]
        y = np.array(b)
        y = np.array(b)/np.max(np.array(b))
        x_fit, y_fit = sine_fitting(x, y)
        ax[ib//gridsize, ib%gridsize].plot(x,y,'.-',color = randcolorlist[ib], label=f"{freq_array[ib]} GHz")
        # ax[ib//gridsize, ib%gridsize].plot(x_fit, y_fit,'-',color = 'grey')
        ax[ib//gridsize, ib%gridsize].set_xlabel("Grid tilt [deg.]")
        ax[ib//gridsize, ib%gridsize].set_ylabel("Beam power [a.u.]")
        ax[ib//gridsize, ib%gridsize].set_yscale("log")
        ax[ib//gridsize, ib%gridsize].legend(loc="upper right")
    plt.show()

    # # show single plot
    # plt.title("Polarization Response of Holography Receiver")
    # plt.xlabel("Grid tilt [deg.]")
    # plt.ylabel("Beam power [a.u.]")
    # for ib, b in enumerate(beams_array):
    #     x = angles_array[ib]
    #     y = np.array(b)
    #     y = np.array(b)/np.max(np.array(b))
    #     x_fit, y_fit = sine_fitting(x, y)
    #     plt.plot(x,y,'.-',color = randcolorlist[ib], label=f"{freq_array[ib]} GHz")
    #     # plt.plot(x_fit, y_fit,'-',color = 'grey')
    #     plt.yscale("log")
    #     plt.legend(loc="upper right")

    # plt.show()

    # # show freq dependence plot
    # x = np.delete(freq_array, -3)
    # y = np.delete(crospratio_array, -3)
    # print(np.mean(y))
    # print(np.mean(y[0:2]))
    # print(np.mean(y[2:]))
    # plt.title("Cross Polarization")
    # plt.xlabel("Frequency [GHz]")
    # plt.ylabel("Cross Pol [%]")
    # plt.plot(x,y,'*-', linewidth=3, markersize=10, color = random_color())
    # plt.show()

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
