"""

Initializing two synthesizers via USB connection for holography experiment.

Grace E. Chesmore
February 2022

"""

import holog_daq
import numpy as np
import usb.core
from holog_daq import synth3

class SynthOpt:
    ENDPOINT_DEC = 2  # , always. according to user manual.
    ENDPOINT_HEX = 0x02
    # Change peak limits depending on desired bin of signal.
    IGNORE_PEAKS_BELOW = int(655)
    IGNORE_PEAKS_ABOVE = int(660)
    F_OFFSET = 10

def set_f(device, freq, lo_id):
    import sys
    import struct
    '''
    Set frequency of synthesizers.
    '''
    # print('Setting frequency to '+str(f)+' MHz')
    n_bytes = 6  # number of bytes remaining in the packet
    # the command number, such as '0x02' for RF output control.
    n_command = 0x01

    if sys.version_info < (3,):  # Python 2?
        def hexfmt(val):
            return '0x{:02X}'.format(ord(val))

    else:
        def hexfmt(val):
            return '0x{:02X}'.format(val)

    print(int(freq * 1.0e6))
    print(struct.pack('>Q', int(freq * 1.0e6)))

    bytes = [hexfmt(b) for b in struct.pack('>Q', int(freq * 1.0e6))]
    print(bytes)
    data = bytearray(64)

    data[0] = SynthOpt.ENDPOINT_HEX
    data[1] = n_bytes
    data[2] = n_command
    i_start = 3

    indx = 0
    while indx < 5:
        data[int(indx + i_start)] = int(bytes[indx + i_start], 16)
        print(data[int(indx + i_start)])
        indx = indx + 1

    lo_id[int(device)].write(SynthOpt.ENDPOINT_DEC, data)

def read_f(device,LO_ID):

    LO_STATE = LO_ID[int(device)]

    FREQ = 0

    return FREQ

N = 18

F_OFFSET = 5  # in MHz
F = int(195.0 * 1000.0 / N)  # MHz

# Contact the synthesizer USB ports
LOs = tuple(usb.core.find(find_all=True, idVendor=0x10C4, idProduct=0x8468))

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
synth3.set_RF_output(0, 1, LOs)  # Turn on the RF output. (device,state)
synth3.set_RF_output(1, 1, LOs)
set_f(0, F, LOs)

read_f(0, LOs)
