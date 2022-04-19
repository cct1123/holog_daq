"""
Polarization Measurement script.
Grace E. Chesmore, Apr 2022
"""
import datetime
import logging
import os,sys
import platform
import time
import casperfpga
import holog_daq
import numpy as np
from holog_daq import fpga_daq3, poco3, synth3

is_py3 = int(platform.python_version_tuple()[0]) == 3 # True if running in python3
SynthOpt = synth3.SynthOpt # Read in synthesizer settings

def get_pol_data(fre, angle):
	'''
	Get the data from the FPGA for a given frequency and angle.
	'''
	now = datetime.datetime.now() # get the current time
	today = str(now.day) + "-" + str(now.month) + "-" + str(now.year) # get the date

	N_MULT = 12 # source multiplication factor
	F_START = int(fre * 1000.0 / N_MULT)  # start frequency in MHz
	SynthOpt.F_OFFSET = 10  # offset frequency in MHz

	SynthOpt.IGNORE_PEAKS_BELOW = int(986) # ignore peaks below this frequency
	SynthOpt.IGNORE_PEAKS_ABOVE = int(990) # ignore peaks above this frequency
	# SynthOpt.IGNORE_PEAKS_BELOW = int(655)
	# SynthOpt.IGNORE_PEAKS_ABOVE = int(660)

	# Define some wait times
	DELTA_T_USB_CMD = 0.5
	T_BETWEEN_DELTA_F = 0.5
	T_BETWEEN_SAMP_TO_AVG = 0.5
	T_TO_MOVE_STAGE = 1
	DELTA_T_VELMEX_CMD = 0.25

	# Define the number of samples to average
	fpga_daq3.RoachOpt.N_CHANNELS = 21
	nsamp = int(1)

	STR_FILE_OUT = (
	"../Data/pol_" + str(fre) + "GHz_" + str(angle) + "deg_" + today + ".txt"
	) # output file name

	arr2D_all_data = np.zeros(
	(nsamp, (4 * fpga_daq3.RoachOpt.N_CHANNELS + 5))
	)  
	# where the 7 extra are f,x,y,phi,... 
	# x_cur,y_cur, index_signal of peak cross power 
	# in a single bin (where phase is to be measured)

	def MakePolData(f, LOs, baseline, fpga):
		'''
		Get the data for a given frequency and angle from the FPGA.
		'''
		# Because we're at the center of the map:
		x = 0
		y = 0
		print("begin MakeBeamMap() for f = " + str(f))
		synth3.set_f(0, f, LOs) # set the synthesizer frequency LO1
		synth3.set_f(1, f + SynthOpt.F_OFFSET, LOs) # set the synthesizer frequency LO2

		# Initialize arrays
		arr_phase = np.zeros((fpga_daq3.RoachOpt.N_CHANNELS, 1))
		arr_aa = np.zeros((fpga_daq3.RoachOpt.N_CHANNELS, 1))
		arr_bb = np.zeros((fpga_daq3.RoachOpt.N_CHANNELS, 1))
		arr_ab = np.zeros((fpga_daq3.RoachOpt.N_CHANNELS, 1))
		index_signal = 0

		phi = angle # polarization angle
		print(
		    " Recording data: f: "
		    + str(f)
		    + " angle: ("
		    + str(int(angle))
		    + ") degs"
		) 

		arr_aa, arr_bb, arr_ab, arr_phase, index_signal = fpga_daq3.TakeAvgData(
		    baseline, fpga, SynthOpt
		) # get the data

		arr2D_all_data[0] = np.array(
		    [f]
		    + [x]
		    + [y]
		    + [phi]
		    + [index_signal]
		    + arr_aa.tolist()
		    + arr_bb.tolist()
		    + arr_ab.tolist()
		    + arr_phase.tolist(),dtype=object
		) # save the data

	# START OF MAIN:
	fpga = None
	roach, opts, baseline = fpga_daq3.roach2_init() # initialize the FPGA settings

	if __name__ == "__main__":

	    loggers = [] 
	    lh = poco3.DebugLogHandler() 
	    logger = logging.getLogger(roach) 
	    logger.addHandler(lh) 
	    logger.setLevel(10)

	try:
		########### Setting up ROACH Connection ###############################
		print("------------------------")
		print("Programming FPGA with call to a python2 prog...")
		# basically starting a whole new terminal and running this script
		err = os.system("/opt/anaconda2/bin/python2 upload_fpga_py2.py") # program the FPGA in Python2

		print("Connecting to server %s ... " % (roach)),
		if is_py3:
		    fpga = casperfpga.CasperFpga(roach) # connect to the FPGA Python3
		else:
		    fpga = casperfpga.katcp_fpga.KatcpFpga(roach) # connect to the FPGA Python2
		time.sleep(1) # wait for the connection to be made

		if fpga.is_connected(): # check if the connection was made
		    print("ok\n") 
		else:
		    print("ERROR connecting to server %s.\n" % (roach)) # if not, print an error message
		    poco3.exit_fail(fpga) # exit the program
		######################################################################

		LOs = synth3.get_LOs() # get the synthesizer settings
		synth3.set_RF_output(0, 1, LOs) # turn on the RF output
		synth3.set_RF_output(1, 1, LOs) # turn on the RF output

		f_sample = F_START 
		print("Begining map where frequency = " + str(fre) + "GHz.")
		time.sleep(T_BETWEEN_DELTA_F) 
		# Now is time to take a beam map
		MakePolData(f_sample, LOs, baseline, fpga) # get the data
		print("Beam Map Complete.")

		arr2D_all_data = np.around(arr2D_all_data, decimals=3) # round the data
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
		) # save the data to txt file

	except KeyboardInterrupt:
	    poco3.exit_clean(fpga) # exit the program
	except:
	    poco3.exit_fail(fpga, lh) # exit the program

	return STR_FILE_OUT # return the file name

F_test = 150 # GHz
angle_test = 0 # deg
get_pol_data(F_test,angle_test) # get the data