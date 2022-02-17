"""

Initializing two synthesizers via USB connection for holography experiment.

Grace E. Chesmore
February 2022

"""

import usb.core
import numpy as np
# import holog_daq
# from holog_daq import synth
import sys
import struct
import synth3

N = 18

F_OFFSET = 5  # in MHz
F = int(180.0 * 1000.0 / N)  # MHz

# Contact the synthesizer USB ports
LOs = tuple(usb.core.find(find_all=True, idVendor=0x10C4, idProduct=0x8468))
print(LOs)
print(LOs[0].bus, LOs[0].address)
print(LOs[1].bus, LOs[1].address)

if (LOs[0] is None) or (LOs[1] is None):
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

# Set the frequency of the RF output, in MHz. (device, state).
# You must have the device's RF output in state (1) before doing this.
synth3.set_RF_output(0, 0, LOs)  # Turn on the RF output. (device,state)
synth3.set_RF_output(1, 0, LOs)
synth3.set_f(0, F, LOs)
