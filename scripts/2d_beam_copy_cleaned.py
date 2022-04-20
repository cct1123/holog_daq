"""
Beam Mapping script.
Scans source in XY plane and records amplitude and phase at each point.
Grace E. Chesmore, Feb 2022

All stage commands are commented out to make this adaptable for other stage setups.
"""
import datetime
import logging
import os
import platform
import time

import casperfpga
import holog_daq
import numpy as np
from holog_daq import fpga_daq3, poco3, synth3

is_py3 = int(platform.python_version_tuple()[0]) == 3

# import xy_agent.xy_connect as stage
# stage_xy = stage.XY_Stage.latrt_xy_stage()
# stage_xy.enable()

steps_per_cm = 1574.80316  # [steps/cm conversion]

SynthOpt = synth3.SynthOpt


def beam2d(fre, angle, step, label):

    # Begin with source centered above the LATRt window.
    # Then define this spot to the stages as (0,0) "center".

    # stage_xy.position = [0,0]

    now = datetime.datetime.now()
    today = str(now.day) + "-" + str(now.month) + "-" + str(now.year)

    N_MULT = 12
    F_START = int(fre * 1000.0 / N_MULT)  # in MHz
    F_STOP = F_START
    SynthOpt.F_OFFSET = 10  # in MHz
    freq = F_STOP

    SynthOpt.IGNORE_PEAKS_BELOW = int(986)
    SynthOpt.IGNORE_PEAKS_ABOVE = int(990)
    # SynthOpt.IGNORE_PEAKS_BELOW = int(655)
    # SynthOpt.IGNORE_PEAKS_ABOVE = int(660)

    DELTA_T_USB_CMD = 0.5
    T_BETWEEN_DELTA_F = 0.5
    T_BETWEEN_SAMP_TO_AVG = 0.5
    T_TO_MOVE_STAGE = 1
    DELTA_T_VELMEX_CMD = 0.25

    DELTA_X_Y = step * np.round(steps_per_cm) / steps_per_cm
    X_MIN_ANGLE = -angle * DELTA_X_Y / step
    X_MAX_ANGLE = angle * DELTA_X_Y / step
    Y_MIN_ANGLE = -angle * DELTA_X_Y / step
    Y_MAX_ANGLE = angle * DELTA_X_Y / step
    PHI_MIN_ANGLE = 0
    PHI_MAX_ANGLE = 0
    DELTA_PHI = 90

    fpga_daq3.RoachOpt.N_CHANNELS = 21

    N_PTS = (X_MAX_ANGLE - X_MIN_ANGLE) / DELTA_X_Y + 1

    prodX = prodY = prodPHI = 0
    if X_MIN_ANGLE == X_MAX_ANGLE:
        prodX = 1
    else:
        prodX = int(abs(X_MAX_ANGLE - X_MIN_ANGLE) / DELTA_X_Y + 1)
    if Y_MIN_ANGLE == Y_MAX_ANGLE:
        prodY = 1
    else:
        prodY = int(abs(Y_MAX_ANGLE - Y_MIN_ANGLE) / DELTA_X_Y + 1)
    if PHI_MIN_ANGLE == PHI_MAX_ANGLE:
        prodPHI = 1
    else:
        prodPHI = 1  # int(abs(PHI_MAX_ANGLE - PHI_MIN_ANGLE)/DELTA_PHI + 1)

    nsamp = int(prodX * prodY * prodPHI)

    print("nsamp = " + str(nsamp))
    STR_FILE_OUT = (
        "Data/" + str(fre) + "GHz_" + str(angle) + "deg_2D_" + label + today + ".txt"
    )
    arr2D_all_data = np.zeros(
        (nsamp, (4 * fpga_daq3.RoachOpt.N_CHANNELS + 7))
    )  # , where the 7 extra are f,x,y,phi,... x_cur,y_cur, index_signal of peak cross power in a single bin (where phase is to be measured)

    def MakeBeamMap(i_f, f, LOs, baseline, fpga):
        i = 0
        print("begin MakeBeamMap() for f = " + str(f))
        synth3.set_f(0, f, LOs)
        synth3.set_f(1, f + SynthOpt.F_OFFSET, LOs)

        arr_phase = np.zeros((fpga_daq3.RoachOpt.N_CHANNELS, 1))
        arr_aa = np.zeros((fpga_daq3.RoachOpt.N_CHANNELS, 1))
        arr_bb = np.zeros((fpga_daq3.RoachOpt.N_CHANNELS, 1))
        arr_ab = np.zeros((fpga_daq3.RoachOpt.N_CHANNELS, 1))
        index_signal = 0

        print(" move x to minimum angle")
        if X_MIN_ANGLE != 0:
            # stage_xy.move_x_cm(X_MIN_ANGLE,1) # Move motor 1 (our x axis) by X_MIN_ANGLE degrees.
            # stage_xy.wait()
            time.sleep(0.5)
        print(" move y to minimum angle")
        if Y_MIN_ANGLE != 0:
            # stage_xy.move_y_cm(Y_MIN_ANGLE,1) # Move motor 1 (our y axis) by Y_MIN_ANGLE degrees.
            # stage_xy.wait()
            time.sleep(0.5)
        for y in np.linspace(Y_MIN_ANGLE, Y_MAX_ANGLE, int(N_PTS)):
            for x in np.linspace(X_MIN_ANGLE, X_MAX_ANGLE, int(N_PTS)):

                # Read out current x y position
                # position = np.array(stage_xy.position)
                # x_cur = position[0]
                # y_cur = position[1]

                phi = 90
                print(
                    " Recording data: f: "
                    + str(f)
                    + "), x: ("
                    + str(int(x))
                    + "/"
                    + str(int(X_MAX_ANGLE))
                    + "), y: ("
                    + str(int(y))
                    + "/"
                    + str(int(Y_MAX_ANGLE))
                    + ")"
                )
                arr_aa, arr_bb, arr_ab, arr_phase, index_signal = fpga_daq3.TakeAvgData(
                    baseline, fpga, SynthOpt
                )
                arr2D_all_data[i] = (
                    [f]
                    + [x]
                    + [y]
                    + [x]
                    + [y]
                    + [phi]
                    + [index_signal]
                    + arr_aa.tolist()
                    + arr_bb.tolist()
                    + arr_ab.tolist()
                    + arr_phase.tolist()
                )
                i = i + 1
                print(str(i) + "/" + str(nsamp))

                if x < X_MAX_ANGLE:
                    print("moving x forward")
                    if abs(DELTA_X_Y) != 0:
                        # stage_xy.move_x_cm(DELTA_X_Y,1) #Move motor 1 (our x axis) by X_MIN_ANGLE degrees.
                        # stage_xy.wait()
                        time.sleep(0.5)

            print("moving x backward.")
            if abs(X_MAX_ANGLE - X_MIN_ANGLE) != 0:
                # pos = stage_xy.position
                # desired_x_pos = X_MIN_ANGLE
                # move_x = pos[0]-desired_x_pos
                # stage_xy.move_x_cm(-move_x,1) #Move motor 1 (our x axis) by X_MIN_ANGLE degrees.
                # stage_xy.wait()
                time.sleep(0.5)
            if y < Y_MAX_ANGLE:
                print("moving y forward")
                # stage_xy.move_y_cm(DELTA_X_Y,1) #Move motor 1 (our x axis) by X_MIN_ANGLE degrees.
                # stage_xy.wait()
                time.sleep(0.5)

        print(" returning y home")
        if abs(Y_MAX_ANGLE) != 0:
            # pos = stage_xy.position
            # desired_y_pos = 0
            # move_y = pos[1]-desired_y_pos
            # stage_xy.move_y_cm(-move_y,1) #Move motor 1 (our x axis) by X_MIN_ANGLE degrees.
            # stage_xy.wait()
            time.sleep(0.5)
        print(" returning x home")
        if abs(X_MAX_ANGLE) != 0:
            # pos = stage_xy.position
            # desired_x_pos = 0
            # move_x = pos[0]-desired_x_pos
            # stage_xy.move_x_cm(-move_x,1) #Move motor 1 (our x axis) by X_MIN_ANGLE degrees.
            # stage_xy.wait()
            time.sleep(0.5)

        print(" end f = " + str(f))

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
        # if not opts.skip:
            # basically starting a whole new terminal and running this script
        err = os.system("/opt/anaconda2/bin/python2 upload_fpga_py2.py")
            # assert err == 0
        # else:
        #     print("Skipped.")

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

        LOs = synth3.get_LOs()
        # Turn on the RF output. (device,state)
        synth3.set_RF_output(0, 1, LOs)
        synth3.set_RF_output(1, 1, LOs)

        f_sample = F_START
        print("Begining map where frequency = " + str(fre) + "GHz.")
        time.sleep(T_BETWEEN_DELTA_F)
        # Now is time to take a beam map
        MakeBeamMap(0, f_sample, LOs, baseline, fpga)
        print("Beam Map Complete.")

        arr2D_all_data = np.around(arr2D_all_data, decimals=3)
        print("Saving data...")
        np.savetxt(
            STR_FILE_OUT,
            arr2D_all_data,
            fmt="%.3e",
            header=(
                "f_sample(GHz), x, y, phi, index_signal of peak cross power, and "
                + str(fpga_daq3.RoachOpt.N_CHANNELS)
                + " points of all of following: aa, bb, ab, phase (deg.)"
            ),
        )

        # print('Done. Back at position: (%d,%d).' %(stage_xy.position[0],stage_xy.position[1]))

    except KeyboardInterrupt:
        poco3.exit_clean(fpga)
    except:
        poco3.exit_fail(fpga, lh)

    return STR_FILE_OUT


span = 3
res = 1
str_out = beam2d(130, span, res, "co_")

# stage_xy.disable()

fpga_daq3.plot_data(str_out)
