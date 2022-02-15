"""
Initialize CASPER ROACH2 FPGA channels.
Grace Chesmore
Modified February 2022
"""
import logging
import sys
import time

import casperfpga
import numpy as np

import fpga_daq
import poco

katcp_port = 7147
# File to program the FPGA
bitstream = "t4_roach2_noquant_fftsat.fpg"

if __name__ == "__main__":
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage("poco_init.py")
    p.set_description(__doc__)
    # Here is where we can change integration time
    p.add_option(
        "-l",
        "--acc_len",
        dest="acc_len",
        type="int",
        default=0.5
        * (2 ** 28)
        / 2048,  # for low pass filter and amplifier this seems like a good value, though not tested with sig. gen. # 25 jan 2018: 0.01
        help="Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048.",
    )  # for roach full setup.
    p.add_option(
        "-g",
        "--gain",
        dest="gain",
        type="int",
        default=2,
        help="Set the digital gain (4bit quantisation scalar). default is 2.",
    )
    p.add_option(
        "-s",
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip reprogramming the FPGA and configuring EQ.",
    )
    p.add_option(
        "-f",
        "--fpg",
        dest="fpgfile",
        type="str",
        default="",
        help="Specify the bof file to load",
    )
    opts, args = p.parse_args(sys.argv[1:])
    roach = "192.168.4.20"

    if opts.fpgfile != "":
        bitstream = opts.fpgfile
    else:
        bitstream = bitstream

try:

    loggers = []
    lh = poco.DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print("Connecting to server %s ... " % (roach)),
    fpga = casperfpga.katcp_fpga.KatcpFpga(roach)
    # Old version of this command:
    # fpga = casperfpga.CasperFpga(roach)
    time.sleep(1)

    if fpga.is_connected():
        print("ok\n")
    else:
        print("ERROR connecting to server %s.\n" % (roach))
        fpga_daq.exit_fail(fpga, lh)

    print("------------------------")
    print("Programming FPGA...")
    if not opts.skip:
        sys.stdout.flush()
        fpga.upload_to_ram_and_program(bitstream)
        time.sleep(10)
        print("done")
    else:
        print("Skipped.")

    print("Configuring fft_shift...")
    fpga.write_int("fft_shift", (2 ** 32) - 1)
    print("done")

    print("Configuring accumulation period...")
    fpga.write_int("acc_len", opts.acc_len)
    print("done")

    print("Resetting board, software triggering and resetting error counters...")
    fpga.write_int("ctrl", 0)
    fpga.write_int("ctrl", 1 << 17)  # arm
    fpga.write_int("ctrl", 0)
    fpga.write_int("ctrl", 1 << 18)  # software trigger
    fpga.write_int("ctrl", 0)
    fpga.write_int("ctrl", 1 << 18)  # issue a second trigger
    print("done")

    print("flushing channels...")
    for chan in range(1024):
        sys.stdout.flush()
    print("done")

    print("All set up. Try plotting using plot_cross_phase_no_quant.py")

except KeyboardInterrupt:
    fpga_daq.exit_clean(fpga)

fpga_daq.exit_clean(fpga)
