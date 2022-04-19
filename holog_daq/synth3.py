"""

Functions for operating two synthesizers via USB connection for holography experiment.

Grace E. Chesmore
February 2022

"""
import struct
import sys

import numpy as np
import usb.core


class SynthOpt:
    ENDPOINT_DEC = 2  # , always. according to user manual.
    ENDPOINT_HEX = 0x02
    # Change peak limits depending on desired bin of signal.
    # IGNORE_PEAKS_BELOW = int(655)
    # IGNORE_PEAKS_ABOVE = int(660)
    IGNORE_PEAKS_BELOW = int(738)
    IGNORE_PEAKS_ABOVE = int(740)
    F_OFFSET = 5


def set_RF_output(device, state, lo_id):
    '''
    Turn synthesizers on or off.
    For state, e.g. '1' for command '0x02' will turn ON the RF output.
    '''
    print("Setting RF output")
    n_bytes = 2  # number of bytes remaining in the packet
    # the command number, such as '0x02' for RF output control.
    n_command = 0x02
    data = bytearray(64)
    data[
        0
    ] = (
        SynthOpt.ENDPOINT_HEX
    )
    # I do think this has to be included and here,
    # because excluding SynthOpt.ENDPOINT as data[0]
    # makes the synth not change its draw of current.
    data[1] = n_bytes
    data[2] = n_command
    data[3] = state

    lo_id[int(device)].write(SynthOpt.ENDPOINT_DEC, data)


def set_f(device, freq, lo_id):
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

    bytes = [hexfmt(b) for b in struct.pack('>Q', int(freq * 1.0e6))]

    data = bytearray(64)
    data[0] = SynthOpt.ENDPOINT_HEX
    data[1] = n_bytes
    data[2] = n_command
    i_start = 3

    indx = 0
    while indx < 5:
        data[int(indx + i_start)] = int(bytes[indx + i_start], 16)
        # print(data[int(indx+i_start)])
        indx = indx + 1

    lo_id[int(device)].write(SynthOpt.ENDPOINT_DEC, data)


def reset_RF(device, lo_id):
    '''
    Reset synthesizers.
    '''
    print("Resetting RF")
    n_bytes = 2  # number of bytes remaining in the packet
    # the command number, such as '0x02' for RF output control.
    n_command = 0x03
    data = bytearray(64)
    data[0] = SynthOpt.ENDPOINT_HEX
    data[1] = n_bytes
    data[2] = n_command
    data[3] = 0x00  # state
    lo_id[int(device)].write(SynthOpt.ENDPOINT_DEC, data)


def get_LOs():
    '''
    Connect to synthesizers.
    '''
    lo_id = tuple(usb.core.find(
        find_all=True, idVendor=0x10C4, idProduct=0x8468))
    print(lo_id[0].bus, lo_id[0].address)
    print(lo_id[1].bus, lo_id[1].address)
    if (lo_id[0] is None) or (lo_id[1] is None):  # Was device found?
        raise ValueError("Device not found.")
    else:
        print(str(np.size(lo_id)) + " device(s) found:")

    indx = 0
    while indx < np.size(lo_id):  # Make sure the USB device is ready to receive commands
        lo_id[indx].reset()
        reattach = False
        if lo_id[indx].is_kernel_driver_active(0):
            reattach = True
            lo_id[indx].detach_kernel_driver(0)
        lo_id[indx].set_configuration()
        indx = indx + 1
    return lo_id


def read_f(device, lo_id):
    '''
    Read current frequency state of synthesizers.
    '''
    print(lo_id[int(device)])
    print(lo_id[int(device)].read(SynthOpt.ENDPOINT_DEC))
