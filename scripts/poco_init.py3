"""
Initialize connection with ROACH2 FPGA and flush+re-start channels.
"""
# from __future__ import print_function
import logging
import os

# check if you're in python 2 or 3
import platform
import time

import casperfpga
import holog_daq
from holog_daq import fpga_daq3, poco3

is_py3 = int(platform.python_version_tuple()[0]) == 3

fpga = None
roach, opts, baseline = fpga_daq3.roach2_init()

try:
    loggers = []
    lh = poco3.DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print("------------------------")
    print("Programming FPGA with call to a python2 prog...")
    # if not opts.skip:
        # basically starting a whole new terminal and running this script
    err = os.system("/opt/anaconda2/bin/python2 upload_fpga_py2.py")
    assert err == 0
    # else:
        # print("Skipped.")

    print("Connecting to server %s ... " % (roach))
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

except KeyboardInterrupt:
    poco3.exit_clean(fpga)
except:
    poco3.exit_fail(fpga)
