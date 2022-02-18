# !/usr/bin/env python
from __future__ import print_function
import casperfpga
import time
import sys
import os
import logging

import platform

assert int(platform.python_version_tuple()[0]) == 2


def exit_fail(fpga, lh):
    """
    Returns fail and prints log entries.
    """
    print("FAILURE DETECTED. Log entries:\n", lh.printMessages())
    try:
        fpga.stop()
    except:
        pass
    raise
    exit()


def exit_clean(fpga):
    """
    Stops FPGA function upon exiting.
    """
    try:
        fpga.stop()
    except:
        pass
    exit()


class DebugLogHandler(logging.Handler):
    """A logger for KATCP tests."""

    def __init__(self, max_len=100):
        """Create a TestLogHandler.
            @param max_len Integer: The maximum number of log entries
                                    to store. After this, will wrap.
        """
        logging.Handler.__init__(self)
        self._max_len = max_len
        self._records = []

    def emit(self, record):
        """Handle the arrival of a log message."""
        if len(self._records) >= self._max_len:
            self._records.pop(0)
        self._records.append(record)

    def clear(self):
        """Clear the list of remembered logs."""
        self._records = []

    def setMaxLen(self, max_len):
        self._max_len = max_len

    def printMessages(self):
        for i in self._records:
            if i.exc_info:
                print("%s: %s Exception: " % (i.name, i.msg), i.exc_info[0:-1])
            else:
                if i.levelno < logging.WARNING:

                    print("%s: %s" % (i.name, i.msg))

                elif (i.levelno >= logging.WARNING) and (i.levelno < logging.ERROR):
                    print("%s: %s" % (i.name, i.msg))
                elif i.levelno >= logging.ERROR:
                    print("%s: %s" % (i.name, i.msg))
                else:
                    print("%s: %s" % (i.name, i.msg))


katcp_port = 7147
DEFAULT_FPG = "t4_roach2_noquant_fftsat.fpg"
DEFAULT_ROACH = "192.168.4.20"

if __name__ == "__main__":
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage("poco_init_no_quant.py")
    p.set_description(__doc__)
    # here is where we can change integration time
    p.add_option(
        "-l",
        "--acc_len",
        dest="acc_len",
        type="int",
        default=0.5
        * (2 ** 28)
        // 2048,  # for low pass filter and amplifier this seems like a good value, though not tested with sig. gen. #	25 jan 2018: 0.01
        help="Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048.",
    )  # for roach full setup.
    p.add_option(
        "-f",
        "--fpg",
        dest="fpgfile",
        type="str",
        default=DEFAULT_FPG,
        help="Specify the bof file to load",
    )
    p.add_option("--roach", default=DEFAULT_ROACH)
    opts, args = p.parse_args(sys.argv[1:])
    roach = opts.roach

    if not os.path.exists(opts.fpgfile):
        p.error("fpgfile does not exist: %s" % opts.fpgfile)

try:

    loggers = []
    lh = DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print("Connecting to server %s ... " % (roach)),
    fpga = casperfpga.katcp_fpga.KatcpFpga(roach)
    # fpga = casperfpga.CasperFpga(roach)
    time.sleep(1)

    if fpga.is_connected():
        print("ok\n")
    else:
        print("ERROR connecting to server %s.\n" % (roach))
        exit_fail(fpga, lh)

    print("------------------------")
    print("Programming FPGA...")
    sys.stdout.flush()
    fpga.upload_to_ram_and_program(opts.fpgfile)
    print(" ... now wait a bit")
    time.sleep(10)
    print("done")

    print("Configuring fft_shift...")
    fpga.write_int("fft_shift", (2 ** 32) - 1)
    print("done")

    print("Configuring accumulation period...")
    fpga.write_int("acc_len", int(opts.acc_len))

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
    exit_clean(fpga)

exit_clean(fpga)
