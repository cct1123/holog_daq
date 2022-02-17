# from __future__ import print_function

import casperfpga
import time
import numpy
import struct
import logging
import numpy as np
import datetime
import usb.core
import usb.util
import datetime
import getpass
#import matplotlib
import sys
import os

# check if you're in python 2 or 3
import platform
is_py3 = int(platform.python_version_tuple()[0]) == 3

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

steps_per_cm = 1574.80316

now = datetime.datetime.now()
today = str(now.day) + '-' +str(now.month) + '-'+str(now.year)

def running_mean(x, N):
    return numpy.convolve(x, numpy.ones((N,))/N)[(N-1):]

N_MULT = 12
NREP = 1
L_MEAN = 1
N_TO_AVG = 1
N_CHANNELS=21

fre =130
F_START= int(fre*1000.//N_MULT) #in MHz
F_STOP = F_START
F_OFFSET = 10 #in MHz

# IGNORE_PEAKS_BELOW = int(655)
# IGNORE_PEAKS_ABOVE = int(660)
IGNORE_PEAKS_BELOW = int(986)
IGNORE_PEAKS_ABOVE = int(990)
ENDPOINT_DEC=2 #, always. according to syntonic user manual.
ENDPOINT_HEX=0x02

T_BETWEEN_DELTA_F = 0.5
DELTA_T_USB_CMD = 0.5
T_BETWEEN_SAMP_TO_AVG = 0.5
T_TO_MOVE_STAGE = 1
DELTA_T_VELMEX_CMD = 0.25

step = .5 # cm
DELTA_X_Y = step*np.round(steps_per_cm)/steps_per_cm

Y_MIN = 0
Y_MAX =  0
X_MIN = -25*DELTA_X_Y/step
X_MAX = -X_MIN
angle = X_MAX
PHI_MIN =0
PHI_MAX = 0 #90.

 #Make this an odd number. When odd, it is the total number of ROACH channels that will get recorded, including the peak signal, which will be in middle.
prodX = prodY = prodPHI = 0
if X_MIN == X_MAX:
    prodX = 1
else:
    prodX = int(abs(X_MAX - X_MIN)/DELTA_X_Y +1)
if Y_MIN == Y_MAX:
    prodY = 1
else:
    prodY = int(abs(Y_MAX - Y_MIN)/DELTA_X_Y +1)
if PHI_MIN == PHI_MAX:
    prodPHI = 1
else:
    prodPHI = 1 #int(abs(PHI_MAX_ANGLE - PHI_MIN_ANGLE)/DELTA_PHI + 1)

nfreq = int(abs(F_START - F_STOP)*10 + 1)
nsamp = int( prodX * prodY * prodPHI )
print('nsamp = '+str(nsamp))
STR_FILE_OUT = '../Data/'+str(str(fre)+'GHz_'+str(int(angle))+'deg_Hcut_'+today+'.txt')
arr2D_all_data=numpy.zeros((nsamp,(4*N_CHANNELS+7)))#, where the 5 extra are f,x,y,phi, index_signal of peak cross power in a single bin (where phase is to be measured)
print(STR_FILE_OUT )
REGISTER_LO_1 = 5000 #Labjack register number for the Labjack DAC0 output, which goes to LO_1.
REGISTER_LO_2 = 5002 #Labjack register number for the Labjack DAC1 output, which goes to LO_2.

##################################################################
# 	Roach Definitions
##################################################################
F_CLOCK_MHZ = 500
f_max_MHz = (F_CLOCK_MHZ/4)
KATCP_PORT=7147


def exit_fail():
    print('FAILURE DETECTED. Log entries:\n',lh.printMessages())
    try:
        fpga.stop()
    except: pass
    raise
    exit()

def exit_clean():
    try:
        fpga.stop()
    except: pass
    exit()

def get_data(baseline):
    #print('   start get_data function')
    acc_n = fpga.read_uint('acc_num')
    print('   Grabbing integration number %i'%acc_n)

    #get cross_correlation data...
    a_0r=struct.unpack('>512l',fpga.read('dir_x0_%s_real'%baseline,2048,0))
    a_1r=struct.unpack('>512l',fpga.read('dir_x1_%s_real'%baseline,2048,0))
    a_0i=struct.unpack('>512l',fpga.read('dir_x0_%s_imag'%baseline,2048,0))
    a_1i=struct.unpack('>512l',fpga.read('dir_x1_%s_imag'%baseline,2048,0))
    b_0i=struct.unpack('>512l',fpga.read('dir_x0_%s_imag'%baseline,2048,0))
    b_1i=struct.unpack('>512l',fpga.read('dir_x1_%s_imag'%baseline,2048,0))
    b_0r=struct.unpack('>512l',fpga.read('dir_x0_%s_real'%baseline,2048,0))
    b_1r=struct.unpack('>512l',fpga.read('dir_x1_%s_real'%baseline,2048,0))
    interleave_cross_a=[]
    interleave_cross_b=[]

    #get auto correlation data (JUST the A, B inputs)...
    a_0=struct.unpack('>512l',fpga.read('dir_x0_bb_real',2048,0))
    a_1=struct.unpack('>512l',fpga.read('dir_x1_bb_real',2048,0))
    b_0=struct.unpack('>512l',fpga.read('dir_x0_dd_real',2048,0))
    b_1=struct.unpack('>512l',fpga.read('dir_x1_dd_real',2048,0))
    interleave_auto_a=[]
    interleave_auto_b=[]


    #interleave cross-correlation and auto correlation data.
    for i in range(512):
        #cross
        interleave_cross_a.append(complex(a_0r[i], a_0i[i]))
        interleave_cross_a.append(complex(a_1r[i], a_1i[i]))
        interleave_cross_b.append(complex(b_0r[i], b_0i[i]))#For phase, new, test.
        interleave_cross_b.append(complex(b_1r[i], b_1i[i]))#For phase, new, test

        #auto
        interleave_auto_a.append(a_0[i])#'interleave' even and odd timestreams back into the original timestream (b.c. sampling rate is 2x your FPGA clock).
        interleave_auto_a.append(a_1[i])
        interleave_auto_b.append(b_0[i])#'interleave' even and odd timestreams back into the original timestream (b.c. sampling rate is 2x your FPGA clock).
        interleave_auto_b.append(b_1[i])

    #print('   end get_data function')
    return acc_n,interleave_cross_a,interleave_cross_b,interleave_auto_a,interleave_auto_b

def drawDataCallback(baseline):
    #print('running get_data  function from drawDataCallback')
    acc_n,interleave_cross_a,interleave_cross_b,interleave_auto_a,interleave_auto_b= get_data(baseline)
    val=running_mean(numpy.abs(interleave_cross_a),L_MEAN)
    val[int(IGNORE_PEAKS_ABOVE):]=0
    val[: int(IGNORE_PEAKS_BELOW)]=0
    arr_index_signal = numpy.argpartition(val, -2)[-2:]
    index_signal = arr_index_signal[1]
    # IS THIS NECESSARY? Probably not here, at least. freq = numpy.linspace(0,f_max_MHz,len(numpy.abs(interleave_cross_a)))
    arr_ab = (numpy.abs(interleave_cross_a))
    arr_phase = (180./numpy.pi)*numpy.unwrap((numpy.angle(interleave_cross_b)))
    phase_signal = arr_phase[index_signal]
    arr_aa = (numpy.abs(interleave_auto_a))
    arr_bb = (numpy.abs(interleave_auto_b))

    #Only record relevant channels, right around peak:
    arr_aa = arr_aa[(index_signal - (N_CHANNELS//2)) : (1+index_signal + (N_CHANNELS//2))]
    arr_bb = arr_bb[(index_signal - (N_CHANNELS//2)) : (1+index_signal + (N_CHANNELS//2))]
    arr_ab = arr_ab[(index_signal - (N_CHANNELS//2)) : (1+index_signal + (N_CHANNELS//2))]
    arr_phase = arr_phase[(index_signal - (N_CHANNELS//2)) : (1+index_signal + (N_CHANNELS//2))]

    return running_mean(arr_aa,L_MEAN),running_mean(arr_bb,L_MEAN),running_mean(arr_ab,L_MEAN), arr_phase, index_signal

def TakeAvgData():
    arr_phase= numpy.zeros((N_CHANNELS,1))
    arr_aa= numpy.zeros((N_CHANNELS,1))
    arr_bb= numpy.zeros((N_CHANNELS,1))
    arr_ab= numpy.zeros((N_CHANNELS,1))
    arr_index =numpy.zeros((1,1))

    arr2D_phase= numpy.zeros((N_TO_AVG,N_CHANNELS))#array of phase data, which I will take the mean of
    arr2D_aa=numpy.zeros((N_TO_AVG,N_CHANNELS))
    arr2D_bb=numpy.zeros((N_TO_AVG,N_CHANNELS))
    arr2D_ab=numpy.zeros((N_TO_AVG,N_CHANNELS))
    arr2D_index= numpy.zeros((N_TO_AVG,1))
    j = 0
    #to average according to each unique index of peak signal, rather than taking mean at the mean value of index of peak signal
    arr_var_unique =numpy.zeros((N_TO_AVG,1)) # A test to see how well we can use mean data over 'N_TO_AVG'.
    while (j < N_TO_AVG):
        #print('In TakeAvgData(), j = ('+str(j)+'/N_TO_AVG)'+' and we are about to drawDataCallback')
        arr2D_aa[j],arr2D_bb[j],arr2D_ab[j],arr2D_phase[j], arr2D_index[j]=drawDataCallback(baseline)
        #^^^^take in data from the roach. see function "drawDataCallback" above for how this works. "arr2D" array take in data across all frequency bins of the roach.
        j = j+1

    arr_phase=arr2D_phase.mean(axis=0)
    arr_aa=arr2D_aa.mean(axis=0)
    arr_bb=arr2D_bb.mean(axis=0)
    arr_ab=arr2D_ab.mean(axis=0)
    arr_index=arr2D_index.mean(axis=0)

    return arr_aa, arr_bb, arr_ab, arr_phase, arr_index

def MakeBeamMap(i_f, f):
    i=0
    print('begin MakeBeamMap() for f = '+str(f))

    arr_phase= numpy.zeros((N_CHANNELS,1))
    arr_aa= numpy.zeros((N_CHANNELS,1))
    arr_bb= numpy.zeros((N_CHANNELS,1))
    arr_ab= numpy.zeros((N_CHANNELS,1))
    index_signal = 0

    print(' move x to minimum angle')

    for x in numpy.arange (X_MIN,X_MAX+DELTA_X_Y,DELTA_X_Y):
        for y in numpy.arange(Y_MIN,Y_MAX+DELTA_X_Y, DELTA_X_Y):
            phi = 90
            arr_aa, arr_bb, arr_ab, arr_phase,index_signal = TakeAvgData()
            arr2D_all_data[i] = ([f]+[x]+[y]+[x]+[y]+[phi]+[index_signal]+arr_aa.tolist()+arr_bb.tolist()+arr_ab.tolist()+arr_phase.tolist())
            i = i+1

# debug log handler
class DebugLogHandler(logging.Handler):
    """A logger for KATCP tests."""

    def __init__(self,max_len=100):
        """Create a TestLogHandler.
            @param max_len Integer: The maximum number of log entries
                                    to store. After this, will wrap.
        """
        logging.Handler.__init__(self)
        self._max_len = max_len
        self._records = []

    def emit(self, record):
        """Handle the arrival of a log message."""
        if len(self._records) >= self._max_len: self._records.pop(0)
        self._records.append(record)

    def clear(self):
        """Clear the list of remembered logs."""
        self._records = []

    def setMaxLen(self,max_len):
        self._max_len=max_len

    def printMessages(self):
        for i in self._records:
            if i.exc_info:
                print('%s: %s Exception: '%(i.name,i.msg),i.exc_info[0:-1])
            else:
                if i.levelno < logging.WARNING:
                    print('%s: %s'%(i.name,i.msg))
                elif (i.levelno >= logging.WARNING) and (i.levelno < logging.ERROR):
                    print('%s: %s'%(i.name,i.msg))
                elif i.levelno >= logging.ERROR:
                    print('%s: %s'%(i.name,i.msg))
                else:
                    print('%s: %s'%(i.name,i.msg))


fpga = None
roach,opts,baseline = roach2_init()
    
try:
    loggers = []
    lh=DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('------------------------')
    print('Programming FPGA with call to a python2 prog...')
    if not opts.skip:
        # basically starting a whole new terminal and running this script
        err = os.system('/opt/anaconda2/bin/python2 upload_fpga_py2.py')
        assert(err==0)
    else:
        print('Skipped.')


    print('Connecting to server %s ... '%(roach)),
    if is_py3:
        fpga = casperfpga.CasperFpga(roach)
    else:
        fpga = casperfpga.katcp_fpga.KatcpFpga(roach)
    time.sleep(1)

    if fpga.is_connected():
        print('ok\n')
    else:
        print('ERROR connecting to server %s.\n'%(roach))
        exit_fail()

    i = 0

    while (i < nfreq):
        f_sample = F_START#(((VfreqSet-1.664)/dv_over_df)*(12.0) + 120.0)
        print('Begining step '+str(i)+' of '+str(nfreq)+', where frequency = '+str(f_sample))
        time.sleep(T_BETWEEN_DELTA_F)
        #Now is time to take a beam map
        MakeBeamMap(i, f_sample)
        i = i+1

    ##
    print('Beam Map Complete.')

    arr2D_all_data = numpy.around(arr2D_all_data,decimals=3)
    print('Saving data...')
    numpy.savetxt(STR_FILE_OUT,arr2D_all_data,fmt='%.3e',header=('f_sample(GHz), x, y, phi, index_signal of peak cross power, and '+str(N_CHANNELS)+' points of all of following: aa, bb, ab, phase (deg.)'))

    print('Done. Exiting.')

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

