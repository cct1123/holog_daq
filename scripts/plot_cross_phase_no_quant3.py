"""
This script demonstrates grabbing data from a wideband Pocket correlator and plotting it using numpy/pylab. 
Designed for use with TUT4 at the 2009 CASPER workshop.

Author: Jason Manley, August 2009.
Modified: Tyrone van Balla, November 2015
Modified: Grace E. Chesmore, October 2020
"""

import array
import datetime
import logging
import os
import struct
import sys
import time

import matplotlib

import casperfpga

matplotlib.use("TkAgg")  # do this before importing pylab

import platform

import matplotlib.pyplot as plt
import numpy as np
import serial
import usb.core
import usb.util

import holog_daq
from holog_daq import fpga_daq3, poco3, synth3

is_py3 = int(platform.python_version_tuple()[0]) == 3

# Added by Charlie 2019-11-04
ylim_lo = 5.0e3
ylim_hi = 5.0e9

xlim_lo = 500
xlim_hi = 1000

# ENDPOINT_DEC=2 #, always. according to user manual.
# ENDPOINT_HEX=0x02
sys.setrecursionlimit(3000)  # by ATJ, to get longer recursion time.
bitstream = "t4_roach2_noquant_fftsat.fpg"
f_clock_MHz = 500
f_max_MHz = f_clock_MHz / 4
katcp_port = 7147
N = 18
F_OFFSET = 5  # 5*f_clock_MHz/500 #MHz
F = int(190.0 * 1000.0 / N)  # MHz


def drawDataCallback(baseline):

    l_mean = 1
    window = 1
    # Set the frequency of the RF output, in MHz.
    # (device, state) You must have the device's
    # RF output in state (1) before doing this.

    synth3.set_f(0, F, LOs)
    synth3.set_f(1, int(F + F_OFFSET), LOs)

    IGNORE_PEAKS_BELOW = int(0)
    IGNORE_PEAKS_ABOVE = int(1090)
    matplotlib.pyplot.clf()
    time.sleep(0.75)
    acc_n, interleave_cross_a, interleave_cross_b, interleave_auto_a, interleave_auto_b = fpga_daq3.get_data(
        baseline, fpga
    )

    freq = np.linspace(0, f_max_MHz, len(np.abs(interleave_cross_a)))
    x_index = np.linspace(0, 1024, len(np.abs(interleave_cross_a)))

    # matplotlib.pyplot.subplot(111)
    which = 0
    val = []
    val_b = []
    arr_index_signal = []

    interleave_auto_a = np.array(interleave_auto_a)
    interleave_auto_b = np.array(interleave_auto_b)
    interleave_cross_a = np.array(interleave_cross_a)
    valaa = fpga_daq3.running_mean(np.abs(interleave_auto_a), l_mean)
    valbb = fpga_daq3.running_mean(np.abs(interleave_auto_b), l_mean)
    valab = fpga_daq3.running_mean(np.abs(interleave_cross_a), l_mean)

    val_copy_i_eval = np.array(valab)
    val_copy_i_eval[int(IGNORE_PEAKS_ABOVE) :] = 0
    val_copy_i_eval[: int(IGNORE_PEAKS_BELOW)] = 0

    matplotlib.pyplot.semilogy(x_index, valaa, color="b", label="aa", alpha=0.5)
    matplotlib.pyplot.semilogy(x_index, valbb, color="r", label="bb", alpha=0.5)
    matplotlib.pyplot.semilogy(x_index, valab, color="g", label="cross")
    matplotlib.pyplot.legend()

    matplotlib.pyplot.ylim(ylim_lo, ylim_hi)
    matplotlib.pyplot.xlim(xlim_lo, xlim_hi)

    arr_index_signal = np.argpartition(val_copy_i_eval, -2)[
        -2:
    ]  # grab the indices of the two largest signals.

    matplotlib.pyplot.grid(which="major")
    matplotlib.pyplot.ylabel("Running Power: Cross")
    matplotlib.pyplot.title("Integration number %i \n%s" % (acc_n, baseline))

    # Find peak cross signal, print value and the freq. at which it occurs
    if (
        (arr_index_signal[1] != 0)
        & (arr_index_signal[1] != 1)
        & (arr_index_signal[1] != 2)
        & (arr_index_signal[1] != 3)
    ):
        index_signal = arr_index_signal[1]

    else:
        index_signal = arr_index_signal[0]

    power_cross = (np.abs(interleave_cross_a))[index_signal]
    arr_phase = np.degrees(np.angle(interleave_cross_a))

    power_auto_a = (np.abs(interleave_auto_a))[index_signal]
    power_auto_b = (np.abs(interleave_auto_b))[index_signal]

    if (index_signal <= window / 2) and index_signal <= 2:
        print(
            "ERROR: your window size is too large. subtracting half of it from index will throw you outside array. Window/2 = "
            + str(window / 2)
            + ", index_signal = "
            + str(index_signal)
        )
    else:
        max_cross_pwr = np.max(
            (np.abs(interleave_cross_a))[IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE]
        )
        max_auto_A_pwr = np.max(
            (np.abs(interleave_auto_a))[IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE]
        )
        max_auto_B_pwr = np.max(
            (np.abs(interleave_auto_b))[IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE]
        )
        index_cross = (
            np.argmax(
                (np.abs(interleave_cross_a))[IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE]
            )
            + IGNORE_PEAKS_BELOW
        )
        index_AA = (
            np.argmax(
                (np.abs(interleave_auto_a))[IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE]
            )
            + IGNORE_PEAKS_BELOW
        )
        index_BB = (
            np.argmax(
                (np.abs(interleave_auto_b))[IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE]
            )
            + IGNORE_PEAKS_BELOW
        )

        max_mean_cross_pwr = np.max(
            (fpga_daq3.running_mean(np.abs(interleave_cross_a), l_mean))[
                IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE
            ]
        )
        max_mean_auto_A_pwr = np.max(
            (fpga_daq3.running_mean(np.abs(interleave_auto_a), l_mean))[
                IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE
            ]
        )
        max_mean_auto_B_pwr = np.max(
            (fpga_daq3.running_mean(np.abs(interleave_auto_b), l_mean))[
                IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE
            ]
        )
        index_mean_cross = (
            np.argmax(
                (fpga_daq3.running_mean(np.abs(interleave_cross_a), l_mean))[
                    IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE
                ]
            )
            + IGNORE_PEAKS_BELOW
        )
        index_mean_AA = (
            np.argmax(
                (fpga_daq3.running_mean(np.abs(interleave_auto_a), l_mean))[
                    IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE
                ]
            )
            + IGNORE_PEAKS_BELOW
        )
        index_mean_BB = (
            np.argmax(
                (fpga_daq3.running_mean(np.abs(interleave_auto_b), l_mean))[
                    IGNORE_PEAKS_BELOW:IGNORE_PEAKS_ABOVE
                ]
            )
            + IGNORE_PEAKS_BELOW
        )

    fig.canvas.manager.window.after(100, drawDataCallback, baseline)
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.ylabel("Running Power: Cross")
    plt.title("Integration number %i \n%s" % (acc_n, baseline))
    matplotlib.pyplot.show()


# START OF MAIN:
fpga = None
roach, opts, baseline = fpga_daq3.roach2_init()

# START OF MAIN:
if __name__ == "__main__":

    loggers = []
    lh = poco3.DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)
    roach, opts, baseline = fpga_daq3.roach2_init()

try:
    ########### Setting up ROACH Connection ##################
    ##########################################################
    print("------------------------")
    print("Programming FPGA with call to a python2 prog...")
    if not opts.skip:
        # basically starting a whole new terminal and running this script
        err = os.system("/opt/anaconda2/bin/python2 upload_fpga_py2.py")
        assert err == 0
    else:
        print("Skipped.")

    print("Connecting to server %s ... " % (roach)),
    if is_py3:
        fpga = casperfpga.CasperFpga(roach)
    else:
        fpga = casperfpga.katcp_fpga.KatcpFpga(roach)
    time.sleep(1)

    if fpga.is_connected():
        print("ok\n")
    else:
        print("ERROR connecting to server %s.\n" % (roach))
        poco3.exit_fail(fpga)
    ##########################################################
    ##########################################################

    ### #prepare synths ###
    LOs = tuple(usb.core.find(find_all=True, idVendor=0x10C4, idProduct=0x8468))
    print("LO1 bus: %d, address: %d" % (LOs[0].bus, LOs[0].address))
    print("LO2 bus: %d, address: %d" % (LOs[1].bus, LOs[1].address))

    if (LOs[0] is None) or (LOs[1] is None):  # Was device found?
        raise ValueError("Device not found.")
    else:
        print(str(np.size(LOs)) + " device(s) found:")

    ii = 0
    while ii < np.size(LOs):
        LOs[ii].reset()
        reattach = False  # Make sure the USB device is ready to receive commands.
        if LOs[ii].is_kernel_driver_active(0):
            reattach = True
            LOs[ii].detach_kernel_driver(0)
        LOs[ii].set_configuration()
        ii = ii + 1

    synth3.set_RF_output(0, 1, LOs)  # Turn on the RF output. (device,state)
    synth3.set_RF_output(1, 1, LOs)
    synth3.set_f(
        0, F, LOs
    )  # Set the frequency of the RF output, in MHz. (device, state) You must have the device's RF output in state (1) before doing this.
    synth3.set_f(1, int(F + F_OFFSET), LOs)
    # set up the figure with a subplot for each polarisation to be plotted
    fig = matplotlib.pyplot.figure()

    # start the process
    fig.canvas.manager.window.after(100, drawDataCallback, baseline)
    matplotlib.pyplot.show()
    print("Plotting complete. Exiting...")


except KeyboardInterrupt:
    poco3.exit_clean(fpga)

poco3.exit_clean(fpga)
