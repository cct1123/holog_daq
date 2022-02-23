"""

Functions used to grab data from a wideband Pocket correlator and plotting it using numpy/pylab.
Designed for use with TUT4 at the 2009 CASPER workshop.

Grace E. Chesmore, August 2021

"""

import logging
import struct
import sys
import time
from optparse import OptionParser

import casperfpga
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import holog_daq
from holog_daq import synth3

plt.style.use("ggplot")


def running_mean(x, N):
    """

    Calculates running mean of 1D array.

    """
    return np.convolve(x, np.ones((N,)) / N)[(N - 1):]


class RoachOpt:

    """

    ROACH2 configuration settings.

    """

    BITSTREAM = "t4_roach2_noquant_fftsat.fpg"

    f_clock_MHz = 500
    f_max_MHz = f_clock_MHz / 4
    N_CHANNELS = 21
    N_TO_AVG = 1
    L_MEAN = 1
    katcp_port = 7147

    ip = "192.168.4.20"


def get_data(baseline, fpga):
    """

    Read out data from ROACH2 FPGA.

    """

    acc_n = fpga.read_uint("acc_num")

    # print('Grabbing integration number %i'%acc_n)

    # get cross_correlation data...

    a_0r = struct.unpack(">512l", fpga.read(
        "dir_x0_%s_real" % baseline, 2048, 0))

    a_1r = struct.unpack(">512l", fpga.read(
        "dir_x1_%s_real" % baseline, 2048, 0))

    a_0i = struct.unpack(">512l", fpga.read(
        "dir_x0_%s_imag" % baseline, 2048, 0))

    a_1i = struct.unpack(">512l", fpga.read(
        "dir_x1_%s_imag" % baseline, 2048, 0))

    b_0i = struct.unpack(">512l", fpga.read(
        "dir_x0_%s_imag" % baseline, 2048, 0))

    b_1i = struct.unpack(">512l", fpga.read(
        "dir_x1_%s_imag" % baseline, 2048, 0))

    b_0r = struct.unpack(">512l", fpga.read(
        "dir_x0_%s_real" % baseline, 2048, 0))

    b_1r = struct.unpack(">512l", fpga.read(
        "dir_x1_%s_real" % baseline, 2048, 0))

    interleave_cross_a = []

    interleave_cross_b = []

    # get auto correlation data (JUST the A, B inputs)...

    a_0 = struct.unpack(">512l", fpga.read("dir_x0_bb_real", 2048, 0))

    a_1 = struct.unpack(">512l", fpga.read("dir_x1_bb_real", 2048, 0))

    b_0 = struct.unpack(">512l", fpga.read("dir_x0_dd_real", 2048, 0))

    b_1 = struct.unpack(">512l", fpga.read("dir_x1_dd_real", 2048, 0))

    interleave_auto_a = []

    interleave_auto_b = []

    # interleave cross-correlation and auto correlation data.
    # interleave cross-correlation and auto correlation data.

    for i in range(512):
        # cross
        interleave_cross_a.append(complex(a_0r[i], a_0i[i]))
        interleave_cross_a.append(complex(a_1r[i], a_1i[i]))
        # For phase, new, test.
        interleave_cross_b.append(complex(b_0r[i], b_0i[i]))
        interleave_cross_b.append(
            complex(b_1r[i], b_1i[i]))  # For phase, new, test

        # auto
        interleave_auto_a.append(a_0[i])
        interleave_auto_a.append(a_1[i])
        interleave_auto_b.append(b_0[i])
        interleave_auto_b.append(b_1[i])

    return (
        acc_n,
        interleave_cross_a,
        interleave_cross_b,
        interleave_auto_a,
        interleave_auto_b,
    )


def roach2_init():
    from optparse import OptionParser
    p = OptionParser()
    # p.set_usage('poco_init_no_quant.py')
    p.set_description(__doc__)
    p.add_option('-s', '--skip', dest='skip', action='store_true',
                 help='Skip reprogramming the FPGA and configuring EQ.')
    p.add_option('-l', '--acc_len', dest='acc_len', type='int', default=.5*(2**28)//2048,
                 # for low pass filter and amplifier this seems like a good value, though not tested with sig. gen.
                 # 25 jan 2018: 0.01
                 help='Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048.')  # for roach full setup.

    p.add_option('-c', '--cross', dest='cross', type='str', default='bd',
                 help='Plot this cross correlation magnitude and phase. default: bd')
    p.add_option('-g', '--gain', dest='gain', type='int', default=2,
                 help='Set the digital gain (4bit quantisation scalar). default is 2.')
    p.add_option('-f', '--fpg', dest='fpgfile', type='str', default='',
                 help='Specify the bof file to load')

    opts, args = p.parse_args(sys.argv[1:])
    roach = '192.168.4.20'
    BIT_S = opts.cross

    return roach, opts, BIT_S


def draw_data_callback(baseline, fpga, syn, LOs, fig, lim):
    """

    Print real-time signal measurement from ROACH.

    """

    l_mean = 1

    window = 1

    # Set the frequency of the RF output, in MHz.

    # (device, state) You must have the device's

    # RF output in state (1) before doing this.

    synth3.set_f(0, syn.F, syn, LOs)

    synth3.set_f(1, int(syn.F + syn.F_OFFSET), syn, LOs)

    IGNORE_PEAKS_BELOW = int(0)

    IGNORE_PEAKS_ABOVE = int(1090)

    matplotlib.pyplot.clf()

    time.sleep(0.75)

    acc_n, interleave_cross_a, interleave_cross_b, interleave_auto_a, interleave_auto_b = get_data(
        baseline, fpga
    )

    freq = np.linspace(0, RoachOpt.f_max_MHz, len(np.abs(interleave_cross_a)))

    x_index = np.linspace(0, 1024, len(np.abs(interleave_cross_a)))

    which = 0

    arr_index_signal = []

    interleave_auto_a = np.array(interleave_auto_a)

    interleave_auto_b = np.array(interleave_auto_b)

    interleave_cross_a = np.array(interleave_cross_a)

    valaa = running_mean(np.abs(interleave_auto_a), l_mean)

    valbb = running_mean(np.abs(interleave_auto_b), l_mean)

    valab = running_mean(np.abs(interleave_cross_a), l_mean)

    val_copy_i_eval = np.array(valab)

    val_copy_i_eval[int(IGNORE_PEAKS_ABOVE):] = 0

    val_copy_i_eval[: int(IGNORE_PEAKS_BELOW)] = 0

    # Here is where we plot the signal.

    matplotlib.pyplot.semilogy(
        x_index, valaa, color="b", label="aa", alpha=0.5)

    matplotlib.pyplot.semilogy(
        x_index, valbb, color="r", label="bb", alpha=0.5)

    plt.semilogy(x_index, valab, color="g", label="cross")

    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

    plt.ylim(lim[0], lim[1])

    plt.xlim(lim[2], lim[3])

    arr_index_signal = np.argpartition(val_copy_i_eval, -2)[-2:]

    # grab the indices of the two largest signals.

    plt.ylabel("Running Power: Cross")

    plt.title("Integration number %i \n%s" % (acc_n, baseline))

    # Find peak cross signal, print value and the freq. at which it occurs

    if (
        arr_index_signal[1] != 0
        and arr_index_signal[1] != 1
        and arr_index_signal[1] != 2
        and arr_index_signal[1] != 3
    ):

        index_signal = arr_index_signal[1]

    else:

        index_signal = arr_index_signal[0]

    power_cross = (np.abs(interleave_cross_a))[index_signal]

    arr_phase = np.degrees(np.angle(interleave_cross_a))

    power_auto_a = (np.abs(interleave_auto_a))[index_signal]

    power_auto_b = (np.abs(interleave_auto_b))[index_signal]

    fig.canvas.manager.window.after(
        100, draw_data_callback, baseline, fpga, syn, LOs, fig, lim
    )

    plt.show()


def drawDataCallback(baseline, fpga, synth_settings):

    acc_n, interleave_cross_a, interleave_cross_b, interleave_auto_a, interleave_auto_b = get_data(
        baseline, fpga)
    val = running_mean(np.abs(interleave_cross_a), RoachOpt.L_MEAN)
    val[int(synth_settings.IGNORE_PEAKS_ABOVE):] = 0
    val[: int(synth_settings.IGNORE_PEAKS_BELOW)] = 0
    arr_index_signal = np.argpartition(val, -2)[-2:]
    index_signal = arr_index_signal[1]

    arr_ab = (np.abs(interleave_cross_a))
    arr_phase = (180./np.pi)*np.unwrap((np.angle(interleave_cross_b)))
    phase_signal = arr_phase[index_signal]
    arr_aa = (np.abs(interleave_auto_a))
    arr_bb = (np.abs(interleave_auto_b))

    # Only record relevant channels, right around peak:
    arr_aa = arr_aa[(index_signal - int(RoachOpt.N_CHANNELS/2))                    : (1+index_signal + int(RoachOpt.N_CHANNELS/2))]
    arr_bb = arr_bb[(index_signal - int(RoachOpt.N_CHANNELS/2))                    : (1+index_signal + int(RoachOpt.N_CHANNELS/2))]
    arr_ab = arr_ab[(index_signal - int(RoachOpt.N_CHANNELS/2))                    : (1+index_signal + int(RoachOpt.N_CHANNELS/2))]
    arr_phase = arr_phase[(index_signal - int(RoachOpt.N_CHANNELS/2))                          : (1+index_signal + int(RoachOpt.N_CHANNELS/2))]

    return running_mean(arr_aa, RoachOpt.L_MEAN), running_mean(arr_bb, RoachOpt.L_MEAN), running_mean(arr_ab, RoachOpt.L_MEAN), arr_phase, index_signal


def TakeAvgData(baseline, fpga, synth_settings):
    arr_phase = np.zeros((RoachOpt.N_CHANNELS, 1))
    arr_aa = np.zeros

    ((RoachOpt.N_CHANNELS, 1))
    arr_bb = np.zeros((RoachOpt.N_CHANNELS, 1))
    arr_ab = np.zeros((RoachOpt.N_CHANNELS, 1))
    arr_index = np.zeros((1, 1))

    # array of phase data, which I will take the mean of
    arr2D_phase = np.zeros((RoachOpt.N_TO_AVG, RoachOpt.N_CHANNELS))
    arr2D_aa = np.zeros((RoachOpt.N_TO_AVG, RoachOpt.N_CHANNELS))
    arr2D_bb = np.zeros((RoachOpt.N_TO_AVG, RoachOpt.N_CHANNELS))
    arr2D_ab = np.zeros((RoachOpt.N_TO_AVG, RoachOpt.N_CHANNELS))
    arr2D_index = np.zeros((RoachOpt.N_TO_AVG, 1))
    j = 0
    # to average according to each unique index of peak signal, rather than taking mean at the mean value of index of peak signal
    # A test to see how well we can use mean data over 'RoachOpt.N_TO_AVG'.
    arr_var_unique = np.zeros((RoachOpt.N_TO_AVG, 1))
    while (j < RoachOpt.N_TO_AVG):
        arr2D_aa[j], arr2D_bb[j], arr2D_ab[j], arr2D_phase[j], arr2D_index[j] = drawDataCallback(
            baseline, fpga, synth_settings)
        j = j+1

    arr_phase = arr2D_phase.mean(axis=0)
    arr_aa = arr2D_aa.mean(axis=0)
    arr_bb = arr2D_bb.mean(axis=0)
    arr_ab = arr2D_ab.mean(axis=0)
    arr_index = arr2D_index.mean(axis=0)

    return arr_aa, arr_bb, arr_ab, arr_phase, arr_index


def plot_data(str_out):
    DATA_1 = np.loadtxt(str_out, skiprows=1)
    DATA = []
    RoachOpt.L_MEAN = 1
    N_INDIV = 7

    line_size = np.size(DATA_1[0])
    nsamp = np.size(DATA_1, 0)
    arr_x = np.zeros(nsamp)
    arr_y = np.zeros(nsamp)
    arr_phi = np.zeros(nsamp)
    amp_cross = np.zeros(nsamp)
    amp_AA = np.zeros(nsamp)
    amp_BB = np.zeros(nsamp)
    amp_var = np.zeros(nsamp)
    phase = np.zeros(nsamp)

    i_AA_begin = int(N_INDIV + (1-1)*(line_size-N_INDIV)/4)
    i_AA_end = int(N_INDIV + (2-1)*(line_size-N_INDIV)/4) - 1
    i_BB_begin = int(N_INDIV + (2-1)*(line_size-N_INDIV)/4)
    i_BB_end = int(N_INDIV + (3-1)*(line_size-N_INDIV)/4) - 1
    i_AB_begin = int(N_INDIV + (3-1)*(line_size-N_INDIV)/4)
    i_AB_end = int(N_INDIV + (4-1)*(line_size-N_INDIV)/4) - 1
    i_phase_begin = int(N_INDIV + (4-1)*(line_size-N_INDIV)/4)
    i_phase_end = int(N_INDIV + (5-1)*(line_size-N_INDIV)/4) - 1

    i = int(0)

    jj = 1
    while (jj <= 1):
        i = int(0)
        if jj == 1:
            str_data = 'Dataset 1'
            DATA = DATA_1
        else:
            DATA = DATA_2
            str_data = 'Dataset 2'
        while (i < (nsamp)):
            # take in raw DATA
            arr_x[i] = DATA[i][1]
            arr_y[i] = DATA[i][2]
            arr_phi[i] = DATA[i][3]
            # use same index singal for both datasets. keep it simple for now.
            index_signal = DATA[i][4]
            arr_AA = np.array(running_mean(
                DATA[i][i_AA_begin: i_AA_end], RoachOpt.L_MEAN))
            arr_BB = np.array(running_mean(
                DATA[i][i_BB_begin: i_BB_end], RoachOpt.L_MEAN))
            arr_AB = np.array(running_mean(
                DATA[i][i_AB_begin: i_AB_end], RoachOpt.L_MEAN))
            arr_phase = np.array(DATA[i][i_phase_begin: i_phase_end])
            n_channels = np.size(arr_AB)

            # make amplitude arrays, in case they need to be plotted.
            amp_cross[i] = np.power(arr_AB[int(n_channels/2)], 1)
            amp_var[i] = np.power(
                np.divide(arr_AB[int(n_channels/2)], arr_AA[int(n_channels/2)]), 2)
            amp_AA[i] = arr_AA[int(n_channels/2)]
            amp_BB[i] = arr_BB[int(n_channels/2)]
            phase[i] = np.remainder(arr_phase[int(n_channels/2)], 360.)
            #print('phase[i] = '+str(phase[i]))
            i = i+1

        amp = amp_var
        amp = np.divide(amp, np.max(amp))
        arr_x = np.unique(arr_x)
        arr_y = np.unique(arr_y)
        X, Y = np.meshgrid(arr_x, arr_y)
        P = amp_cross.reshape(len(arr_x), len(arr_y))
        Z = phase.reshape(len(arr_x), len(arr_y))

        jj = jj + 1

    beam_complex = P * np.exp(Z * np.pi / 180. * np.complex(0, 1))
    #Z = np.transpose(Z)
    #beam_complex = np.transpose(beam_complex)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.pcolormesh(X,Y,Z)
    # plt.title("Phase")
    # plt.xlabel('x (cm.)')
    # plt.ylabel('y (cm.)')
    # plt.colorbar(label = 'Phase')
    # plt.axis("equal")
    # plt.subplot(1,2,2)
    # plt.pcolormesh(X,Y,20*np.log10(abs(beam_complex)/np.max(abs(beam_complex))))
    # plt.title("Power [dB]")
    # plt.xlabel('x (cm.)')
    # plt.ylabel('y (cm.)')
    # plt.colorbar(label = 'Power')
    # plt.axis("equal")
    # plt.show()
