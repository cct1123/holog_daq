"""

Functions used to grab data from a wideband Pocket correlator and plotting it using numpy/pylab.
Designed for use with TUT4 at the 2009 CASPER workshop.

Grace E. Chesmore, August 2021

"""

import struct
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import holog_daq
from holog_daq import synth3

import logging
import casperfpga
from optparse import OptionParser

plt.style.use("ggplot")

def running_mean(x, N):
    """

    Calculates running mean of 1D array.

    """
    return np.convolve(x, np.ones((N,)) / N)[(N - 1) :]

class RoachOpt:

    """

    ROACH2 configuration settings.

    """

    BITSTREAM = "t4_roach2_noquant_fftsat.fpg"

    f_clock_MHz = 500
    f_max_MHz = f_clock_MHz / 4

    katcp_port = 7147

    ip = "192.168.4.20"

def get_data(baseline, fpga):

    """

    Read out data from ROACH2 FPGA.

    """

    acc_n = fpga.read_uint("acc_num")

    # print('Grabbing integration number %i'%acc_n)

    # get cross_correlation data...

    a_0r = struct.unpack(">512l", fpga.read("dir_x0_%s_real" % baseline, 2048, 0))

    a_1r = struct.unpack(">512l", fpga.read("dir_x1_%s_real" % baseline, 2048, 0))

    a_0i = struct.unpack(">512l", fpga.read("dir_x0_%s_imag" % baseline, 2048, 0))

    a_1i = struct.unpack(">512l", fpga.read("dir_x1_%s_imag" % baseline, 2048, 0))

    b_0i = struct.unpack(">512l", fpga.read("dir_x0_%s_imag" % baseline, 2048, 0))

    b_1i = struct.unpack(">512l", fpga.read("dir_x1_%s_imag" % baseline, 2048, 0))

    b_0r = struct.unpack(">512l", fpga.read("dir_x0_%s_real" % baseline, 2048, 0))

    b_1r = struct.unpack(">512l", fpga.read("dir_x1_%s_real" % baseline, 2048, 0))

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
        #cross
        interleave_cross_a.append(complex(a_0r[i], a_0i[i]))
        interleave_cross_a.append(complex(a_1r[i], a_1i[i]))
        interleave_cross_b.append(complex(b_0r[i], b_0i[i]))#For phase, new, test.
        interleave_cross_b.append(complex(b_1r[i], b_1i[i]))#For phase, new, test

        #auto
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
    p.set_usage('poco_init_no_quant.py')
    p.set_description(__doc__)
    p.add_option('-s', '--skip', dest='skip', action='store_true',
        help='Skip reprogramming the FPGA and configuring EQ.')
    p.add_option('-l', '--acc_len', dest='acc_len', type='int',default=.5*(2**28)//2048, #for low pass filter and amplifier this seems like a good value, though not tested with sig. gen. # 25 jan 2018: 0.01
        help='Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048.')#for roach full setup.

    p.add_option('-c', '--cross', dest='cross', type='str',default='bd',
        help='Plot this cross correlation magnitude and phase. default: bd')
    p.add_option('-g', '--gain', dest='gain', type='int',default=2,
        help='Set the digital gain (4bit quantisation scalar). default is 2.')
    p.add_option('-f', '--fpg', dest='fpgfile', type='str', default='',
        help='Specify the bof file to load')

    opts, args = p.parse_args(sys.argv[1:])
    roach='192.168.4.20'
    BIT_S=opts.cross

    return roach,opts,BIT_S

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

    val_copy_i_eval[int(IGNORE_PEAKS_ABOVE) :] = 0

    val_copy_i_eval[: int(IGNORE_PEAKS_BELOW)] = 0

    # Here is where we plot the signal.

    matplotlib.pyplot.semilogy(x_index, valaa, color="b", label="aa", alpha=0.5)

    matplotlib.pyplot.semilogy(x_index, valbb, color="r", label="bb", alpha=0.5)

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
