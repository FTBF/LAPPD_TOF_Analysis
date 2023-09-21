import os
import numpy as np 
import pandas as pd
import bitstruct.c as bitstruct
from matplotlib import pyplot as plt
from matplotlib import colors
import scipy
from scipy.optimize import curve_fit, fsolve, fmin
from scipy.interpolate import splrep, BSpline, CubicSpline
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import uproot
from pylandau import langau_pdf
from time import process_time

avg_fwhms = []

# Some helper functions

def convert_to_list(some_object):
	"""Converts an object into a single-element list if the object is not already a list. Helps when a function works by going through elements of a list, but you want to pass a single element to that function.
	Arguments:
		(any) some_object: what you want to convert to a list (if not already a list)
	"""

	if not isinstance(some_object, list) and not isinstance(some_object, np.ndarray):
		some_object = [some_object]

	return some_object

def find_extrema_spline(xdata, ydata):

	spline_tuple = splrep(xdata, ydata, k=3, s=10000)
	bspline = BSpline(*spline_tuple)
	dbspline = bspline.derivative()
	dcubic_spline = CubicSpline(xdata, dbspline(xdata))
	extrema = dcubic_spline.solve(0, extrapolate=False)

	return extrema, bspline

def compute_sliding_function(xdata, ydata, lbound, rbound, stat_spline, func, slide_increment=1):
	"""Handles the sliding/computing part of finding the autocorrelation function or lag-based chi-squared.
		(ndarray)	xdata:				xdata of the waveform, i.e. the sample times
		(ndarray)	ydata				ydata of the waveform, i.e. the voltages
		(float)		lbound				left bound in the subdomain of xdata you wish to compute the sliding function over
		(float)		rbound				right bound in the subdomain of xdata you wish to compute the sliding function over
		(BSpline)	stat_spline			BSpline object interpolating the subrange of ydata over the subdomain of xdata
		(float)		func				the sliding function you wish to apply. Takes as inputs stationary ydata and slid
										ydata. For autocorrelation returns the integral of stat_ydata*slid_ydata. For chi-squared
										returns integral of (stat_ydata-slid_ydata)**2
		(float)		slide_increment=1	effectively the number of indices you're sliding the data by. Default is 1 (~250 ps), but 
										can be set to less than 1.
	"""

	xdata_sliding = np.copy(xdata)
	func_vals = []
	lags = []
	lag = 0
	lag_factor = 25/256
	while lag < 256:

		xdata_sliding -= lag_factor*slide_increment*np.ones_like(xdata_sliding)
		indices_inbounds = np.linspace(0,255,256,dtype=int)[(xdata_sliding >= lbound) & (xdata_sliding <= rbound)]
		if len(indices_inbounds) == 0:
			lag += slide_increment
			continue

		xdata_shifted_inbounds = xdata_sliding[indices_inbounds]
		ydata_inbounds = ydata[indices_inbounds]
		
		if not indices_inbounds[0] == 0:
			r_ind = indices_inbounds[0]
			l_ind = r_ind - 1
			lbound_yval = np.interp(lbound, [xdata_sliding[l_ind], xdata_sliding[r_ind]], [ydata[l_ind], ydata[r_ind]])
			xdata_shifted_inbounds = np.insert(xdata_shifted_inbounds, 0, lbound)
			ydata_inbounds = np.insert(ydata_inbounds, 0, lbound_yval)

		if not indices_inbounds[-1] == 255:
			l_ind = indices_inbounds[-1]
			r_ind = l_ind + 1
			rbound_yval = np.interp(rbound, [xdata_sliding[l_ind], xdata_sliding[r_ind]], [ydata[l_ind], ydata[r_ind]])
			xdata_shifted_inbounds = np.append(xdata_shifted_inbounds, rbound)
			ydata_inbounds = np.append(ydata_inbounds, rbound_yval)
							
		ydata_stationary = stat_spline(xdata_shifted_inbounds)

		func_vals.append(func(ydata_inbounds, ydata_stationary, xdata_shifted_inbounds))
		lags.append(lag_factor*lag)

		lag += slide_increment

	func_vals = np.array(func_vals)
	lags = np.array(lags)

	return lags, func_vals

#This class represents the ACDC boards, and thus
#in proxy an LAPPD - as in the LAPPD TOF system we plan
#to use one Acdc for each LAPPD, read out in single-ended
#strip mode. This class holds the routines that analyze raw waveforms,
#which are stored in a pandas dataframe that contains information
#related to each channel, including the waveform for each channel. 

#All of the information in the Acdc class stays the same for all events
#except for the waveform info and some metadata, so that is updated 
#on an event by event basis but all else (configs, etc) are kept the same. 

class Acdc:
	def __init__(self, init_data_dict, raw_waveform_data_path_list=None):
		
		# Imports all the initialization data via a dictionary and assigns to instance variables. Right now, I'm
		#	working with a dictionary as input and instance variables rather than a separate calibration object since 
		#	some of the initialization data may need further processing within the Acdc object (e.g. pedestal_data_path -> 
		#	pedestal data via import_raw_data). 
		# 
		# 	Also, some of the following variables may be replaced in function by some of the following variables not yet
		#	used (e.g. voltage_count_curves will eventually replace pedestal_counts as a way to correct ADC data and convert to voltage)
		
		self.station_id = init_data_dict['station_id']		# station position, e.g. '1'
		self.lappd_id = init_data_dict['lappd_id']			# Incom manufacturing number, e.g. '125'
		self.acdc_id = init_data_dict['acdc_id']			# ACDC number (see inventory), e.g. '46'
		self.sync_ch = init_data_dict['sync_ch']			
		self.strip_pos = init_data_dict['strip_pos']		# mm shape=(# channels,); local center positions of each strip relative to bottom of LAPPD
		self.len_cor = init_data_dict['len_cor']			# mm, shape=(# channels,); a correction on the length of the strip + PCB traces per strip

		# I am overriding this as self.sample_times, which I initialize to None. Will need to go back in and figure this stuff out when we know if the init_data_dict is even gonna have an option to sepcify the sample_times
		self.times = init_data_dict['times']				# ps, xxx need better variable name and also better description imo; a list of timebase calibrated times
		self.sample_times = []

		self.wraparound = init_data_dict['wraparound']		# ps, shape=(# channels,);  a constant time associated with the delay for when the VCDL goes from the 255th cap to the 0th cap
		self.vel = init_data_dict['vel']					# mm/ps; average (~500 MHz - 1GHz) propagation velocity of the strip
		self.dt = init_data_dict['dt']						# ps; nominal sampling time interval, 1/(clock to PSEC4 x number of samples)
		_, _, self.pedestal_data = self.import_raw_data(init_data_dict['pedestal_data_path'], is_pedestal_data=True)	# ADC count, shape=(# events, # channels, # capacitors)
		self.pedestal_counts = init_data_dict['pedestal_counts']	# ADC count; a list of 256 integers, which corresponds to each capicitors of VCDL.
		self.pedestal_voltage = init_data_dict['pedestal_voltage']
		self.voltage_count_curves = init_data_dict['voltage_count_curves']
		self.calib_data_file_path = init_data_dict['calib_data_file_path']

		self.cur_times, self.cur_times_320, self.cur_waveforms_raw, self.cur_waveforms = None, None, None, None
		# Imports waveform data if a path was specified upon Acdc initialization
		if raw_waveform_data_path_list is not None:

			# Format of the variables import_raw_data writes to:
			#	self.cur_times: xxx, shape=(# events,)
			# 	self.cur_times_320: xxx, shape=(#events,)
			# 	self.cur_waveforms_raw: # ADC count, shape=(# events, # channels, # capacitors); a list of waveforms (2D arrays) for all events, raw means no processing/pedestal subtraction/voltage conversion
			self.import_raw_data(raw_waveform_data_path_list)

			self.process_raw_data_via_pedestal()
			# self.process_raw_data()

		else:
			print('Initializing ACDC object with no waveform data.')





		# Everything below is leftover, but will kept because a) still need to incorporate some and b) for posterity's sake until I'm sure we can delete

		#"ch": channel number, preferentially matches ACDC data output channel number please. (0, 1, ...)
		#the ACDCs do not have a constant sampling rate, like in self.dt. Instead, each sample
		#has its time relative to the last sample calibrated and stored in a calibration file. 

		#PLEASE VALIDATE!? -JIN- Each capacitor of the VCDL will carry slightly different number of charges even when we feed the entire ring buffer with a 0.0v DC signal.
		#As a result, systemic(non-random) fluctuation is visible at the each sample of raw waveforms. We say that each capacitor has a characteristic 'pedestal' ADC count, which stays effectively constant during an entire analysis.
		#baseline_subtract() function removes the formentioned systemic error from the current waveform by subtracting each pedestal ADC counts from the corresponding samples. 
		#"pedestal_counts": ADC count, a list of 256 integers, which corresponds to each capicitors of VCDL.

		#DO WE NEED 'pedestal_counts'? ISN'T 'voltage_count_curve' A SUPERSET OF 'pedestal_counts'? -JIN-
		#ADC counts do not exactly 'measure' the input voltage, in the sense that each capacitor of the VCDL does not charge completely linearly with the input voltage.
		#Thus the 'voltage-ADC count' curve is measured for each capacitor, and we also consider this as a characteristic curve of the capacitor.
		#voltage_linearization() function reconstructs actual voltage waveform from ADC count waveform utilizing inverse function theorem(??? -JIN)
		#"voltage_count_curves": 256(# of capacitors)*[[voltage, ADC count]*(# of measurement points)], # of measurement points typically being 256.
		#one channel is special, used for synchronization, so we keep it separate 
		self.sync_dict = {"ch": self.sync_ch, "waveform": None, "times": None, "wraparound": None} #similar columns as df, currently hard coding the sync channel as "0"

		#metadata dictionary from the loader of the raw files, holds clock/counter information
		self.cur_times = 0 #s, timestamp of the currently loaded waveform
		self.cur_times_320 = 0 #clock cycles, timestamp of the currently loaded waveform
		self.cur_event_count = 0 #event count of the currently loaded waveform
		self.event_numbers = [] #list of event numbers, in order, for the currently loaded waveform.
		
		# self.calibration_fn = calibration_fn #location of config file
		# commented below out - cameron
		# self.load_calibration() #will load default values if no calibration file provided. clear indicates we want a fresh dataframe

	def import_raw_data(self, raw_data_path_list, is_pedestal_data=False):
		"""Imports binary LAPPD data into ACDC object.
		Arguments:
			(Acdc) self;
			(str) raw_data_path_list - the file path of the binary data;
		"""

		# How many 64 bit words make up the non-header data of a single event
		NUM64BITWORDS = 1440

		# Lists to add data to
		times_320 = []		# the time of each event using the 320 MHz clock
		times = []			# the time of each event using the 1 Hz clock
		data = []			# a 1D array that will be turned into a 3D matrix where one axis gives the event, another axis the channel, and another axis the capacitor

		# CompiledFormat objects, used as quick shortcuts when unpacking data (instead of having to type "u16p48" each time)
		format_time=bitstruct.compile("u64")
		format_header=bitstruct.compile("u16p48")
		format_accheader=bitstruct.compile("u56u8")
		format=bitstruct.compile("u12"*(256*30))

		swapformat="8"*(NUM64BITWORDS)

		# Converts raw_waveform_data_path_list to list if it was mistakenly passed as a string (common error
		#	when working with single binary file).
		raw_data_path_list = convert_to_list(raw_data_path_list)

		# Imports each raw data file in the list supplied
		for raw_data_path in raw_data_path_list:

			# Status update about which data file we're importing
			if is_pedestal_data:
				print(f'Importing pedestal data from \"{raw_data_path}\"')
			else:
				print(f'Importing data from \"{raw_data_path}\"')

			number_of_lines_read = 0
			# Here it actually reads and imports the raw data file
			with open(raw_data_path, "rb") as f:

				line = f.read((1+4+NUM64BITWORDS)*8)

				# This loop breaks when the line length is no longer what we expect it to be (i.e. (1+4+NUM64BITWORDS)*8), which indicates end of file or file is corrupted (missing portions)
				while len(line) == (1+4+NUM64BITWORDS)*8:

					number_of_lines_read+=1

					# get 8 byte acc_header and 8 byte header data from first 16 bytes of data (using byteswap to handle converting endianness)
					acc_header = format_accheader.unpack(bitstruct.byteswap("8", line[0*8:1*8]))
					header = format_header.unpack(bitstruct.byteswap("8", line[1*8:2*8]))

					if acc_header[0] != 0x123456789abcde or header[0] != 0xac9c:
						#print("CORRUPT EVENT!!! ", lnum, "%x"%acc_header[0], "%x"%header[0])
						line = f.read((1+4+NUM64BITWORDS)*8)
						continue
					times_320.extend(format_time.unpack(bitstruct.byteswap("8", line[2*8:3*8])))
					times.extend(format_time.unpack(bitstruct.byteswap("8", line[3*8:4*8])))

					# Gets to actual data and reads it.
					data.extend(format.unpack(bitstruct.byteswap(swapformat, line[5*8:])))

					# Continues the loop
					line = f.read((1+4+NUM64BITWORDS)*8)

		# Turns lists into np arrays and reshapes `data` since `data` is not 1D
		times_320 = np.array(times_320)
		times = np.array(times_320)
		data = np.array(data).reshape([-1,30,256])

		# Immediately overrides object's prior waveform data if it's not pedestal data being imported.
		if not is_pedestal_data:
			
			# See __init__ function for description of the following variables.
			self.cur_times_320, self.cur_times, self.cur_waveforms_raw = times_320, times, data

			self.process_raw_data_via_pedestal()
			# self.process_raw_data()

		# Still returns relevant data
		return times_320, times, data
	
	def process_raw_data_via_pedestal(self):
		"""Cleans up raw waveform data. Current implementation simply subtracts off average pedestal ADC count of each capacitor in each channel. Future implementations will use voltage_count_curves and interpolation to correct ADC counts and convert to voltage.cw_low
			(Acdc) self		
		"""

		# Averages pedestal_data over all the events
		self.pedestal_counts = self.pedestal_data.mean(0)

		# Subtracts pedestal_counts from the waveform data for each event (subtracts by broadcasting the 2D pedestal_counts array to the 3D cur_waveforms_raw array)
		self.cur_waveforms = self.cur_waveforms_raw - self.pedestal_counts
		
		# Have to rearrange channels
		channels = np.array([5,4,3,2,1,0,11,10,9,8,7,6,17,16,15,14,13,12,23,22,21,20,19,18,29,28,27,26,25,24])
		self.cur_waveforms = self.cur_waveforms[:,channels,:].copy()
		self.cur_waveforms_raw = self.cur_waveforms_raw[:,channels,:].copy()

		# Performs wrap-around correction using trigger location
		for i, waveform in enumerate(self.cur_waveforms):

			trigger_low = (((self.cur_times_320[i]+2+2)%8)*32-16)%256

			deltax = np.array([256*4/259]+[256/259]*255)
			deltax = np.roll(deltax, -trigger_low)
			x_data = np.cumsum(deltax)
			self.sample_times.append(x_data*25./256)

			# setting axis is necessary because waveform here is a 2D array
			self.cur_waveforms[i,:,:] = np.roll(waveform.copy(), -trigger_low, axis=1)
			self.cur_waveforms_raw[i,:,:] = np.roll(self.cur_waveforms_raw[i,:,:].copy(), -trigger_low, axis=1)

		self.sample_times = np.array(self.sample_times)

		return
	
	def process_raw_data(self):

		if self.calib_data_file_path is None:
			return
		
		# imports the numpy array from the root file by traversing file tree, reading data as a np array, and reshaping to (#channels,
		# 	#capacitors, #voltages, (voltage, adc))
		in_file = uproot.open(self.calib_data_file_path)
		voltage_counts = np.reshape(in_file["config_tree"]["voltage_count_curves"].array(library="np"), (30,256,256,2))

		time_offsets = np.reshape(in_file["config_tree"]["time_offsets"].array(library="np"), (30,256))

		for thing in time_offsets:
			for value in thing:
				if value == 1e-10:
					print(value)
			print(thing)

		fig, (ax, ax2) = plt.subplots(2)
		channel = 11
		time_offsets_subsample = time_offsets[channel,:]
		samples = np.linspace(0,255,256,dtype=int)
		ax.plot(samples,time_offsets_subsample)
		ax.xaxis.set_ticks_position('both')
		ax.yaxis.set_ticks_position('both')
		ax2.hist(time_offsets_subsample, bins=25)
		plt.minorticks_on()
		plt.show()

		# have to normalize since we get voltage as a value on [0,4096]
		voltage_counts[:,:,:,0] = voltage_counts[:,:,:,0]*1.2/4096.
		
		self.cur_waveforms = np.zeros_like(self.cur_waveforms_raw, dtype=np.float64)

		voltageLin = []
		for ch in range(0, 30):
			voltageLin.append([])
			for cap in range(0, 256):

				single_voltage_counts = voltage_counts[ch, cap, :, :]

				# for each channel and each capacitor, applies the Savitzky-Golay filter to ADC data to smooth it out, returning same data but adjusted with 
				#	the smoothing applied
				single_voltage_counts[:, 1] = scipy.signal.savgol_filter(single_voltage_counts[:, 1], 41, 2)
				voltage_count_sorted = single_voltage_counts[single_voltage_counts[:, 1].argsort()]

				adjusted_values = np.interp(self.cur_waveforms_raw[:,ch,cap], voltage_count_sorted[:,1], voltage_count_sorted[:,0])

				self.cur_waveforms[:, ch, cap] = adjusted_values

				# fig, ax = plt.subplots()

				# ax.plot(voltage_count_sorted[:,1], voltage_count_sorted[:,0])
				
				# plt.show()
		
		# fig, ax = plt.subplots()

		# event, ch = 600, 15
		# normalized_adc_data = self.cur_waveforms_raw[event, ch, :]
		# normalized_adc_data = normalized_adc_data/normalized_adc_data.min()

		# normalized_voltage_data = self.cur_waveforms[event, ch, :]
		# normalized_voltage_data = 

		# plt.show()

		return

	def save_data_npz(self, file_name, directory_path=None):
		"""Saves current sample times, sample times (320 clock), raw waveform, calibrated waveform, and uncalibrated sample times
		xxx		
		"""

		if directory_path is None:
			directory_path = os.path.dirname(os.path.realpath(__file__))

		if directory_path[-1] != r'/':
			directory_path += r'/'

		print(f'Saving data to \'{directory_path + file_name}\'')		
		np.savez(directory_path + file_name, cur_times=np.copy(self.cur_times), cur_times_320=np.copy(self.cur_times_320), cur_waveforms_raw=np.copy(self.cur_waveforms_raw), cur_waveforms=np.copy(self.cur_waveforms), sample_times=np.copy(self.sample_times))

		return
	
	def load_data_npz(self, file_path):

		print(f'Loading data from {file_path}')
		data_array = np.load(file_path)

		self.cur_times = data_array['cur_times']
		self.cur_times_320 = data_array['cur_times_320']
		self.cur_waveforms_raw = data_array['cur_waveforms_raw']
		self.cur_waveforms = data_array['cur_waveforms']
		self.sample_times = data_array['sample_times']

		print('Successfully loaded!')

		return

	def index_to_time(self, index, times):
		"""Implements quick index-to-time using linear interpolation (add more xxx)"""

		def index_to_time_rec(single_index):
			lower_index = int(single_index)
			upper_index = lower_index+1

			if upper_index == len(times):
				return times[-1]
			else:
				slope_denominator = (259./25)*(times[upper_index] - times[lower_index])
				slope = (times[upper_index] - times[lower_index])/round(slope_denominator)
				return times[lower_index] + slope*single_index

		if not isinstance(index, list) and not isinstance(index, np.ndarray):
			return index_to_time_rec(index)
		else:
			new_array = []
			for element in index:
				new_array.append(index_to_time_rec(element))
			new_array = np.array(new_array)
			return new_array		
			
	# not sure how this one is affected by incorporating sample_times and wrap-around fix xxx 
	def hist_single_cap_counts_vs_ped(self, ch, cap):
		"""Plots a histogram of ADC counts for a single capacitor in a channel for all events recorded in the binary file. Also plots a histogram for the pedestal ADC counts of the same capacitor.
		Arguments:
			(int): channel number
			(int): capacitor number		
		"""

		# Calculates the bins for the histogram using the maximum and minimum ADC counts
		single_cap_ped_counts = self.pedestal_data[:, ch, cap]
		ped_bins_left_edge = single_cap_ped_counts.min()
		ped_bins_right_edge = single_cap_ped_counts.max()+1
		ped_bins = np.linspace(ped_bins_left_edge, ped_bins_right_edge, ped_bins_right_edge-ped_bins_left_edge+1)
		print(f'Minimum pedestal ADC count: {ped_bins_left_edge}')
		print(f'Maximum pedestal ADC count: {ped_bins_right_edge-1}')

		# Calculates the bins for the histogram using the maximum and minimum ADC counts
		single_cap_raw_counts = self.cur_waveforms_raw[:, ch, cap]
		raw_bins_left_edge = single_cap_raw_counts.min()
		raw_bins_right_edge = single_cap_raw_counts.max()+1
		raw_bins = np.linspace(raw_bins_left_edge, raw_bins_right_edge, raw_bins_right_edge-raw_bins_left_edge+1)
		print(f'Minimum raw waveform ADC count: {raw_bins_left_edge}')
		print(f'Maximum raw waveform ADC count: {raw_bins_right_edge-1}')

		# Plots histogram
		fig, ax = plt.subplots()
		ax.hist(single_cap_ped_counts, histtype='step', linewidth=3, bins=ped_bins)
		ax.hist(single_cap_raw_counts, histtype='step', linewidth=3, bins=raw_bins)
		ax.set_xlabel('ADC Counts')
		ax.set_ylabel('Number of events (per 1 count bins)')
		ax.set_yscale('log')
		plt.show()

		return
	
	def plot_ped_corrected_pulse(self, event, channels=None):
		"""Plots a single event across multiple channels to compare raw and pedestal-corrected ADC counts.
		Arguments:
			(Acdc) self
			(int) event: the index number of the event you wish to plot
			(int / list) channels: a single channel or list of channels you wish to plot for the event
		"""

		# Checks if user specifies a subset of channels, if so, makes sure subset is of type list, if not, uses all channels.
		if channels is None:
			channels = np.linspace(0, 29, 30, dtype=int)
		channels = convert_to_list(channels)

		# Creates 1D array of x_data (all 256 capacitors) and computes 2D array (one axis channel #, other axis capacitor #) of
		#	corrected and raw ADC data
		y_data_list = self.cur_waveforms[event,channels,:].reshape(len(channels), -1)
		y_data_raw_list = self.cur_waveforms_raw[event,channels,:].reshape(len(channels), -1)


		fig, (ax1, ax2) = plt.subplots(2, 1)

		# Plots the raw waveform data
		for channel, y_data_raw in enumerate(y_data_raw_list):
			ax1.plot(self.sample_times[event], y_data_raw, label="Channel %i"%channel)

		# Plots the corrected waveform data
		for channel, y_data in enumerate(y_data_list):
			ax2.plot(self.sample_times[event], y_data, label='Channel %i'%channel)		

		# Labels the plots, make them look pretty, and displays the plots
		ax1.set_xlabel("Time sample (ns)")
		ax1.set_ylabel("ADC count (raw)")
		ax1.tick_params(right=True, top=True)
		ax2.set_xlabel("Time sample (ns)")
		ax2.set_ylabel("ADC count (ped corrected)")
		ax2.tick_params(right=True, top=True)
		
		fig.tight_layout()
		plt.show()
		return

	def plot_raw_lappd(self, event):

		waveform = self.cur_waveforms[event]

		xdata = np.linspace(0,255,256)
		ydata = np.linspace(0,29,30)

		fig, ax = plt.subplots()

		norm = colors.CenteredNorm()
		ax.pcolormesh(xdata, ydata, waveform, norm=norm, cmap='bwr')

		plt.show()

		return

	def find_event_centers(self, events=None, DEBUG_EVENTS=False, METHOD='langaus', SAVE=False):
		"""xxx add description

		"""

		# If no `events` list is passed, assume we must find centers for all events in cur_waveforms. 
		if events is None:
			events = np.linspace(0, len(self.sample_times)-1, len(self.sample_times), dtype=int)
		else:
			events = convert_to_list(events)


		t1 = process_time()
		centers = []
		pos_saved = []
		num_skipped_waveforms = 0
		for i in events:

			if i%500 == 0:
				print(f'{i} centers calculated...')

			try:

				waveform = np.copy(self.cur_waveforms[i])	
				
				l_pos_x_data = self.sample_times[i]
				largest_ch, bad_channels = self.largest_signal_ch(waveform)
				# largest_ch_OLD = self.largest_signal_ch_old(waveform)
				# print(f'New largest ch: {largest_ch}\nOld largest ch: {largest_ch_OLD}')
				l_pos_y_data = waveform[largest_ch]

				if DEBUG_EVENTS:
					self.plot_ped_corrected_pulse(i, channels=largest_ch)

				############## IN PROGRESS MISTRIGGER CUT ##############
				l_pos_y_data_max = np.absolute(l_pos_y_data).max()
				what_max_should_be = 200
				if l_pos_y_data_max < what_max_should_be:
					print(f'Error with event {i} (likely is just noise)')
					num_skipped_waveforms += 1
					if DEBUG_EVENTS:
						raise
					else:
						continue

				first_val = np.absolute(l_pos_y_data[0])
				# subset_avg = np.absolute(np.average(l_pos_y_data[205:215]))
				# if first_val >= 2.25*subset_avg and first_val >= 0.10*l_pos_y_data_max:
				print(first_val/l_pos_y_data_max)
				if first_val >= 0.15*l_pos_y_data_max:
					print(f'Error with event {i} (likely has incorrect trigger)')
					num_skipped_waveforms += 1
					if DEBUG_EVENTS:
						raise
					else:
						continue
				########################################################
				
				DIAGNOSTIC_DATA = False

				if 'langaus' in METHOD.lower():
					if 'cfd' in METHOD.lower():
						l_pos = self.find_l_pos_langaus(l_pos_x_data, l_pos_y_data, METHOD='cfd', display=DEBUG_EVENTS)
					else:
						METHOD = 'langaus_mpv'
						l_pos = self.find_l_pos_langaus(l_pos_x_data, l_pos_y_data, display=DEBUG_EVENTS)
				# l_pos_autocor_le = self.find_l_pos_autocor_le_subset(l_pos_x_data, l_pos_y_data, display=DEBUG_EVENTS, diagnostic=DIAGNOSTIC_DATA)
				# l_pos_autocor_full = self.find_l_pos_autocor_full(l_pos_x_data, l_pos_y_data, display=DEBUG_EVENTS, diagnostic=DIAGNOSTIC_DATA)
				# l_pos = self.find_l_pos_cfd(l_pos_y_data)
				# l_pos_spline = self.find_l_pos_spline(l_pos_x_data, l_pos_y_data, display=DEBUG_EVENTS)
				# l_pos_chi_squared = self.find_l_pos_chi_squared(l_pos_x_data, l_pos_y_data, display=DEBUG_EVENTS)
				# l_pos = 1.2

				# xxx what should I do with bad channels? maybe set to zero, maybe something else
				# for ch in bad_channels:
				# 	waveform[ch] = np.zeros_like(waveform[ch])
				# t_pos = self.find_t_pos_ls_gauss(waveform, display=DEBUG_EVENTS)
				# t_pos = self.find_t_pos_simple(waveform)
				t_pos = 1.2

				centers.append((i, l_pos, t_pos))

			except:
				print(f'Error with event {i}')
				num_skipped_waveforms += 1
				if DEBUG_EVENTS:
					raise
				else:
					pass
		t2 = process_time()

		print('\n')
		print('--------------- Calculating Centers ---------------')
		print(f'Number of skipped waveforms: {num_skipped_waveforms}')
		print(f'Total number of waveforms: {len(events)}')
		print(f'Percent skipped: {round(100*num_skipped_waveforms/len(events), 2)}%')
		print(f'Time to calculate centers: {round(t2-t1, 2)} sec')
		print('---------------------------------------------------')
						
		centers = np.array(centers)

		if SAVE == True:
			np.save('centers_' + METHOD, centers)

		return centers
	
	def largest_signal_ch(self, waveform):
		"""Small function to retrieve the channel to be used in the find_l_pos functions.
		Arguments:
			(Acdc) self
			(np.array) waveform: a single event's waveform (2D array) from which the optimal channel will be found		
		"""

		ratio_limit = 0.25

		sorted_greatest_to_least = np.flip(np.argsort(np.absolute(waveform).max(axis=1)))
		
		bad_channels = []
		correct_ch = -1
		for ch in sorted_greatest_to_least:
			ydata = waveform[ch]
			max_index = np.absolute(ydata).argmax()
			if max_index <= 1 or max_index >= len(ydata)-2:
				continue
			l_val, c_val, r_val = ydata[max_index-2], ydata[max_index], ydata[max_index+2]
			if l_val/c_val < ratio_limit or r_val/c_val < ratio_limit:
				bad_channels.append(ch)
				continue
			else:
				correct_ch = ch
				break

		if correct_ch < 0:
			raise
				
		return correct_ch, bad_channels
	
	def find_l_pos_autocor_full(self, xdata, ydata, display=False, diagnostic=False):
		"""xxx fill in description
		
		"""

		autocor_lbound = xdata[0]
		autocor_rbound = xdata[-1]

		# Defines the BSpline object for the stationary data
		spline_tuple = splrep(xdata, ydata, k=3, s=10000)
		bspline = BSpline(*spline_tuple)

		def autocor_func(stat_ydata, slid_ydata, xdata_inbounds):
			return trapezoid(stat_ydata*slid_ydata, xdata_inbounds)

		lags, autocor_vals = compute_sliding_function(xdata, ydata, autocor_lbound, autocor_rbound, bspline, autocor_func)
		
		height_cutoff = 0.3*autocor_vals.max()
		distance_between_peaks = 12 				# in units of indices
		peak_region_radius = 12*(25/256)			# in units of ns

		integral_peaks_rough_indices = find_peaks(autocor_vals, height=height_cutoff, distance=distance_between_peaks)[0]
		integral_peaks_rough_times = lags[integral_peaks_rough_indices]

		extremas = []
		splines = []
		domains = []
		for integral_peak_rough in integral_peaks_rough_times:

			integral_peak_region_cut = (lags > (integral_peak_rough-peak_region_radius)) & (lags < (integral_peak_rough+peak_region_radius))

			lags_cut = lags[integral_peak_region_cut]
			integrals_cut = autocor_vals[integral_peak_region_cut]

			spline_tuple = splrep(lags_cut, integrals_cut, k=3, s=10000)
			data_bspline = BSpline(*spline_tuple)
			ddata_bspline = data_bspline.derivative()
			
			peak_region_domain_lower = integral_peak_rough-peak_region_radius
			if peak_region_domain_lower < 0:
				peak_region_domain_lower = 0

			peak_region_domain = np.linspace(peak_region_domain_lower, integral_peak_rough+peak_region_radius, 100)
			dcubic_spline = CubicSpline(peak_region_domain, ddata_bspline(peak_region_domain))

			extrema = dcubic_spline.solve(0)

			extrema = extrema[(extrema > (integral_peak_rough-peak_region_radius+3*(25/256))) & (extrema < (integral_peak_rough+peak_region_radius-3*(25/256)))]

			if len(extrema) > 0:
				extrema = extrema[data_bspline(extrema).argsort()][-1]
			else:
				extrema = integral_peak_rough
			
			extremas.append(extrema)
			splines.append(data_bspline)
			domains.append(peak_region_domain)

		extremas = np.array(extremas)

		if len(extremas) > 1:
			print('Uh oh larger than 1!')
			raise

		if display:
			fig2, ax2 = plt.subplots()
			ax2.scatter(lags, autocor_vals, label='Discrete Autocorrelation', marker='.')
			ax2.axvline(extremas[0], color='orange', label=f'Peak 1 (lag={round(extremas[0], 2)} ns)')
			ax2.plot(domains[0], splines[0](domains[0]), color='pink', label='Peak 1 Spline')
			# ax2.text(12.8,3.17e5, f'$\Delta t = {round(delta_t, 2)}$ ns', fontdict={'size': 16})
			ax2.legend()
			ax2.set_xlabel('Time delay (ns)')
			ax2.set_ylabel('Autocorrelation value')
			plt.show()
				
		return extremas[0]

	def find_l_pos_autocor_le_subset(self, xdata, ydata, display=False, diagnostic=False):
		"""xxx fill in description
		
		"""

		# Possible values are 'gauss', 'langau'
		diagnostic_method = 'gauss'

		# Determines the indices of the peaks in the prompt and reflected pulses
		height_cutoff = -0.6*ydata.max()
		distance_between_peaks = 20		# in units of indices
		peak_region_radius = 15			# in units of indices
		peaks_rough = find_peaks(-1*ydata, height=height_cutoff, distance=distance_between_peaks)[0]
		prompt_peak_index, reflect_peak_index = np.sort(peaks_rough[ydata[peaks_rough].argsort()[0:2]])
	
		# Creates subregions of data around the reflect peak
		reflect_lbound = reflect_peak_index - int((reflect_peak_index-prompt_peak_index)/2)-5 # lower bound is a bit left of the midway between peaks
		reflect_ubound = reflect_peak_index + 6
		ydata_subrange = ydata[reflect_lbound:reflect_ubound]
		subdomain = xdata[reflect_lbound:reflect_ubound]
		peak_region_lower, peak_region_upper = xdata[reflect_peak_index-3], xdata[reflect_peak_index+3]

		# Solves for the extrema of the reflect peak
		spline_tuple = splrep(subdomain, ydata_subrange, k=3, s=10000)
		bspline = BSpline(*spline_tuple)
		dbspline = bspline.derivative()
		reflect_cspline = CubicSpline(subdomain, bspline(subdomain))
		dcubic_spline = CubicSpline(subdomain, dbspline(subdomain))
		extrema = dcubic_spline.solve(0, extrapolate=False)
		reflect_peak_max = reflect_cspline(extrema[(extrema > peak_region_lower) & (extrema < peak_region_upper)])	# finds the extrema that is near our original find_peaks value
		reflect_peak_max = reflect_peak_max[0]
		reflect_peak_min_val = reflect_cspline(extrema[0]) + 0.1*(reflect_peak_max - reflect_cspline(extrema[0]))

		# repeating the spline for the prompt peak now
		prompt_lbound = prompt_peak_index - 14
		prompt_ubound = prompt_peak_index + 4
		prompt_subrange = ydata[prompt_lbound:prompt_ubound]
		prompt_subdomain = xdata[prompt_lbound:prompt_ubound]
		prompt_tuple = splrep(prompt_subdomain, prompt_subrange, k=3, s=10000)
		prompt_bspline = BSpline(*prompt_tuple)
		prompt_dbspline = prompt_bspline.derivative()
		prompt_cspline = CubicSpline(prompt_subdomain, prompt_bspline(prompt_subdomain))
		prompt_dcspline = CubicSpline(prompt_subdomain, prompt_dbspline(prompt_subdomain))
		prompt_extrema = prompt_dcspline.solve(0)
		peak_region_lower, peak_region_upper = xdata[prompt_peak_index-3], xdata[prompt_peak_index+3]
		prompt_peak_max = prompt_cspline(prompt_extrema[(prompt_extrema > peak_region_lower) & (prompt_extrema < peak_region_upper)])
		prompt_peak_max = prompt_peak_max[0]

		# Computes the integral bounds
		integral_lower_bound = prompt_cspline.solve(reflect_peak_min_val, extrapolate=False)[0]
		integral_upper_bound = prompt_cspline.solve(0.9*prompt_peak_max, extrapolate=False)[0]

		if display:
			fig3, ax3 = plt.subplots()
			ax3.scatter(xdata, ydata, marker='.', label='Raw data')

			reflect_peak_spline_domain = np.linspace(xdata[reflect_lbound], xdata[reflect_ubound-1], 100)
			ax3.plot(reflect_peak_spline_domain, reflect_cspline(reflect_peak_spline_domain), color='orange')

			prompt_peak_spline_domain = np.linspace(xdata[prompt_lbound], xdata[prompt_ubound-1], 100)
			ax3.plot(prompt_peak_spline_domain, prompt_cspline(prompt_peak_spline_domain), color='green')

			ax3.axhline(reflect_peak_min_val, color='pink', label=f'{round(100*reflect_peak_min_val/reflect_peak_max, 2)}% of reflected peak max')
			ax3.axvline(integral_lower_bound, color='red', label=f'Lower bound ({round(100*reflect_peak_min_val/prompt_peak_max, 2)}% of \nprompt peak max)')
			ax3.axvline(integral_upper_bound, color='purple', label='Upper bound (90% of \nprompt peak max)')
			ax3.legend()
			ax3.set_xlabel('Sample')
			ax3.set_ylabel('ADC Count')
			ax3.set_title('Integration bounds for the autocorrelation function')
			plt.show()

		def autocor_func(stat_ydata, slid_ydata, xdata_inbounds):
			return trapezoid(stat_ydata*slid_ydata, xdata_inbounds)
		
		lags, autocor_vals = compute_sliding_function(xdata, ydata, integral_lower_bound, integral_upper_bound, prompt_bspline, autocor_func)

		height_cutoff = 0.6*autocor_vals.max()
		distance_between_peaks = 5				# in units of indices
		peak_region_radius = 5*(25/256)			# in units of ns

		if diagnostic:
			large_region_radius = 15*(25/256)		# in units of ns
			if diagnostic_method == 'gauss':
				def diag_func(x, N, sigma, mu):
						return (N/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(x-mu)*(x-mu)/(sigma*sigma))
			elif diagnostic_method == 'langau':
				def diag_func(x, A, mu, xi, sigma):
					return A*langau_pdf(x, mu, xi, sigma)
			diag_funcs = []

		integral_peaks_rough_indices = find_peaks(autocor_vals, height=height_cutoff, distance=distance_between_peaks)[0]
		integral_peaks_rough_times = lags[integral_peaks_rough_indices]

		extremas = []
		splines = []
		domains = []
		for integral_peak_rough in integral_peaks_rough_times:

			integral_peak_region_cut = (lags > (integral_peak_rough-peak_region_radius)) & (lags < (integral_peak_rough+peak_region_radius))

			lags_cut = lags[integral_peak_region_cut]
			integrals_cut = autocor_vals[integral_peak_region_cut]

			spline_tuple = splrep(lags_cut, integrals_cut, k=3, s=10000)
			data_bspline = BSpline(*spline_tuple)
			ddata_bspline = data_bspline.derivative()
			
			peak_region_domain_lower = integral_peak_rough-peak_region_radius
			if peak_region_domain_lower < 0:
				peak_region_domain_lower = 0

			peak_region_domain = np.linspace(peak_region_domain_lower, integral_peak_rough+peak_region_radius, 100)
			dcubic_spline = CubicSpline(peak_region_domain, ddata_bspline(peak_region_domain))

			extrema = dcubic_spline.solve(0)

			extrema = extrema[(extrema > (integral_peak_rough-peak_region_radius+3*(25/256))) & (extrema < (integral_peak_rough+peak_region_radius-3*(25/256)))]

			if len(extrema) > 0:
				extrema = extrema[data_bspline(extrema).argsort()][-1]
			else:
				extrema = integral_peak_rough
			
			extremas.append(extrema)
			splines.append(data_bspline)
			domains.append(peak_region_domain)

			if diagnostic:
				integral_peak_region_cut_diag = (lags > (integral_peak_rough-large_region_radius)) & (lags < (integral_peak_rough+large_region_radius))

				integrals_cut_diag = autocor_vals[integral_peak_region_cut_diag]
				lags_cut_diag = lags[integral_peak_region_cut_diag]

				if diagnostic_method == 'gauss':
					N0 = 0.75*integrals_cut_diag.max()
					sigma0 = 0.3*(lags_cut_diag[-1] - lags_cut_diag[0])
					mu0 = lags_cut_diag[integrals_cut_diag.argmax()]
					p0 = [N0, sigma0, mu0]
				elif diagnostic_method == 'langau':
					A0 = 0.2*integrals_cut_diag.max()
					mu0 = lags_cut_diag[integrals_cut_diag.argmax()]
					xi0 = 0.1*(lags_cut_diag[-1] - lags_cut_diag[0])
					sigma0 = 0.1*(lags_cut_diag[-1] - lags_cut_diag[0])
					p0 = [A0, mu0, xi0, sigma0]

				popt, pcov = curve_fit(diag_func, lags_cut_diag, integrals_cut_diag, p0=p0)
				diag_funcs.append(popt)
				if display:
					fig, ax = plt.subplots()
					ax.scatter(lags_cut_diag, integrals_cut_diag, marker='.', label='Raw data')
					diag_fit_domain = np.linspace(lags_cut_diag[0], lags_cut_diag[-1], 200)
					ax.plot(diag_fit_domain, diag_func(diag_fit_domain, *p0), label='p0')
					ax.plot(diag_fit_domain, diag_func(diag_fit_domain, *popt), label='popt')
					ax.set_xlabel('Lag time (ns)')
					ax.legend()
					plt.show()		

		extremas = np.array(extremas)

		delta_t = extremas[1] - extremas[0]

		if display:
			fig2, ax2 = plt.subplots()
			ax2.scatter(lags, autocor_vals, label='Discrete Autocorrelation', marker='.')
			ax2.axvline(extremas[0], color='orange', label=f'Peak 1 (lag={round(extremas[0], 2)} ns)')
			ax2.axvline(extremas[1], color='green', label=f'Peak 2 (lag={round(extremas[1], 2)} ns)')
			ax2.plot(domains[0], splines[0](domains[0]), color='pink', label='Peak 1 Spline')
			ax2.plot(domains[1], splines[1](domains[1]), color='red', label='Peak 2 Spline')
			ax2.text(12.8,3.17e5, f'$\Delta t = {round(delta_t, 2)}$ ns', fontdict={'size': 16})
			ax2.legend()
			ax2.set_xlabel('Time delay (ns)')
			ax2.set_ylabel('Autocorrelation value')
			plt.show()
		
		if diagnostic:
			individual_fwhms = []
			if diagnostic_method == 'gauss':
				fwhm1 = 2.355*diag_funcs[0][1]
				fwhm2 = 2.355*diag_funcs[1][1]
				avg_fwhm = 0.5*(fwhm1 + fwhm2)
				fwhm_start1 = diag_funcs[0][2] - 0.5*fwhm1
				fwhm_start2 = diag_funcs[1][2] - 0.5*fwhm2
				individual_fwhms.append((fwhm1, fwhm_start1))
				individual_fwhms.append((fwhm2, fwhm_start2))
			elif diagnostic_method == 'langau':
				fwhms = []
				whole_domain = np.linspace(lags[0], lags[-1], 5000)
				for popt in diag_funcs:
					y_vals = diag_func(whole_domain, *popt)
					above_hm_lags = whole_domain[y_vals > 0.5*y_vals.max()]
					fwhm = above_hm_lags[-1] - above_hm_lags[0]
					fwhms.append(fwhm)
				avg_fwhm = 0.5*(fwhms[0] + fwhms[1])
			avg_fwhms.append(avg_fwhm)
			if display:
				fig, ax = plt.subplots()
				fig.set_size_inches([10.5, 8])
				ax.scatter(lags, autocor_vals, label='Discrete Autocorrelation', marker='.')
				whole_domain = np.linspace(-5, lags[-1], 6000)
				diag_func1_yvals = diag_func(whole_domain, *diag_funcs[0])
				ax.plot(whole_domain, diag_func1_yvals, color='pink', label='Peak 1 Fit')
				diag_func2_yvals = diag_func(whole_domain, *diag_funcs[1])
				ax.plot(whole_domain, diag_func2_yvals, color='red', label='Peak 2 Fit')
				ax.legend()
				ax.set_xlabel('Time delay (ns)', fontdict={'size': 15})
				ax.set_ylabel('Autocorrelation value', fontdict={'size': 15})
				plt.show()

		return delta_t

	def find_l_pos_chi_squared(self, xdata, ydata, display=False):
		"""xxx fill in description
		
		"""

		# Determines the indices of the peaks in the prompt and reflected pulses
		height_cutoff = -0.6*ydata.max()
		distance_between_peaks = 20		# in units of indices
		peak_region_radius = 15			# in units of indices
		peaks_rough = find_peaks(-1*ydata, height=height_cutoff, distance=distance_between_peaks)[0]
		prompt_peak_index, reflect_peak_index = np.sort(peaks_rough[ydata[peaks_rough].argsort()[0:2]])
	
		# Creates subregions of data around the reflect peak
		reflect_lbound = reflect_peak_index - int((reflect_peak_index-prompt_peak_index)/2)-5 # lower bound is a bit left of the midway between peaks
		reflect_ubound = reflect_peak_index + 6
		ydata_subrange = ydata[reflect_lbound:reflect_ubound]
		subdomain = xdata[reflect_lbound:reflect_ubound]
		peak_region_lower, peak_region_upper = xdata[reflect_peak_index-3], xdata[reflect_peak_index+3]

		# Solves for the extrema of the reflect peak
		spline_tuple = splrep(subdomain, ydata_subrange, k=3, s=10000)
		bspline = BSpline(*spline_tuple)
		dbspline = bspline.derivative()
		reflect_cspline = CubicSpline(subdomain, bspline(subdomain))
		dcubic_spline = CubicSpline(subdomain, dbspline(subdomain))
		extrema = dcubic_spline.solve(0, extrapolate=False)
		reflect_peak_max = reflect_cspline(extrema[(extrema > peak_region_lower) & (extrema < peak_region_upper)])	# finds the extrema that is near our original find_peaks value
		reflect_peak_max = reflect_peak_max[0]
		reflect_peak_min_val = reflect_cspline(extrema[0]) + 0.1*(reflect_peak_max - reflect_cspline(extrema[0]))

		# repeating the spline for the prompt peak now
		prompt_lbound = prompt_peak_index - 14
		prompt_ubound = prompt_peak_index + 4
		prompt_subrange = ydata[prompt_lbound:prompt_ubound]
		prompt_subdomain = xdata[prompt_lbound:prompt_ubound]
		prompt_tuple = splrep(prompt_subdomain, prompt_subrange, k=3, s=10000)
		prompt_bspline = BSpline(*prompt_tuple)
		prompt_dbspline = prompt_bspline.derivative()
		prompt_cspline = CubicSpline(prompt_subdomain, prompt_bspline(prompt_subdomain))
		prompt_dcspline = CubicSpline(prompt_subdomain, prompt_dbspline(prompt_subdomain))
		prompt_extrema = prompt_dcspline.solve(0)
		peak_region_lower, peak_region_upper = xdata[prompt_peak_index-3], xdata[prompt_peak_index+3]
		prompt_peak_max = prompt_cspline(prompt_extrema[(prompt_extrema > peak_region_lower) & (prompt_extrema < peak_region_upper)])
		prompt_peak_max = prompt_peak_max[0]

		# Computes the integral bounds
		integral_lower_bound = prompt_cspline.solve(reflect_peak_min_val, extrapolate=False)[0]
		integral_upper_bound = prompt_cspline.solve(0.9*prompt_peak_max, extrapolate=False)[0]

		if display:
			fig3, ax3 = plt.subplots()
			ax3.scatter(xdata, ydata, marker='.', label='Raw data')

			reflect_peak_spline_domain = np.linspace(xdata[reflect_lbound], xdata[reflect_ubound-1], 100)
			ax3.plot(reflect_peak_spline_domain, reflect_cspline(reflect_peak_spline_domain), color='orange')

			prompt_peak_spline_domain = np.linspace(xdata[prompt_lbound], xdata[prompt_ubound-1], 100)
			ax3.plot(prompt_peak_spline_domain, prompt_cspline(prompt_peak_spline_domain), color='green')

			ax3.axhline(reflect_peak_min_val, color='pink', label=f'{round(100*reflect_peak_min_val/reflect_peak_max, 2)}% of reflected peak max')
			ax3.axvline(integral_lower_bound, color='red', label=f'Lower bound ({round(100*reflect_peak_min_val/prompt_peak_max, 2)}% of \nprompt peak max)')
			ax3.axvline(integral_upper_bound, color='purple', label='Upper bound (90% of \nprompt peak max)')
			ax3.legend()
			ax3.set_xlabel('Sample')
			ax3.set_ylabel('ADC Count')
			ax3.set_title('Integration bounds for the autocorrelation function')
			plt.show()

		def chi2_func(stat_ydata, slid_ydata, xdata_slid):
			return trapezoid((slid_ydata - stat_ydata)*(slid_ydata - stat_ydata), xdata_slid)
		
		lags, chi2_vals = compute_sliding_function(xdata, ydata, integral_lower_bound, integral_upper_bound, prompt_bspline, chi2_func)

		height_cutoff = -0.45*chi2_vals.max()
		distance_between_peaks = 10				# in units of indices
		peak_region_radius = 5*(25/256)			# in units of ns

		integral_peaks_rough_indices = find_peaks(-chi2_vals, height=height_cutoff, distance=distance_between_peaks)[0]
		sorted_integral_peaks_rough_indices = integral_peaks_rough_indices[chi2_vals[integral_peaks_rough_indices].argsort()]
		integral_peak_rough_index = sorted_integral_peaks_rough_indices[0]
		if integral_peak_rough_index == 0:
			integral_peak_rough_index = sorted_integral_peaks_rough_indices[1]

		integral_peak_rough = lags[integral_peak_rough_index]

		integral_peak_region_cut = (lags > (integral_peak_rough-peak_region_radius)) & (lags < (integral_peak_rough+peak_region_radius))

		lags_cut = lags[integral_peak_region_cut]
		integrals_cut = chi2_vals[integral_peak_region_cut]

		spline_tuple = splrep(lags_cut, integrals_cut, k=3, s=10000)
		data_bspline = BSpline(*spline_tuple)
		ddata_bspline = data_bspline.derivative()
		
		peak_region_domain_lower = integral_peak_rough-peak_region_radius
		if peak_region_domain_lower < 0:
			peak_region_domain_lower = 0

		peak_region_domain = np.linspace(peak_region_domain_lower, integral_peak_rough+peak_region_radius, 100)
		dcubic_spline = CubicSpline(peak_region_domain, ddata_bspline(peak_region_domain))

		extrema = dcubic_spline.solve(0)

		extrema = extrema[(extrema > (integral_peak_rough-peak_region_radius+3*(25/256))) & (extrema < (integral_peak_rough+peak_region_radius-3*(25/256)))]

		if len(extrema) > 0:
			extrema = extrema[data_bspline(extrema).argsort()][-1]
		else:
			extrema = integral_peak_rough
		
		# the second extrema for the chi-squared curve is by definition at t=0, so we don't have to subtract anything to get delta_t
		delta_t = extrema

		if display:
			fig2, ax2 = plt.subplots()
			ax2.scatter(lags, chi2_vals, label='Chi-squared', marker='.')
			ax2.axvline(extrema, color='orange', label=f'Peak')
			ax2.plot(peak_region_domain, data_bspline(peak_region_domain), color='red', label='Peak Spline')
			ax2.text(8.2, 2e5, f'$\Delta t = {round(delta_t, 2)}$ ns', fontdict={'size': 16})
			ax2.legend()
			ax2.set_xlabel('Time delay (ns)')
			ax2.set_ylabel('Chi-squared value')
			plt.show()

		return delta_t

	def find_l_pos_langaus(self, xdata, ydata, METHOD='mpv', display=False):

		# A = scale factor, mu = most probable value (maximum), sigma = gaussian standard dev, xi = area under curver
		def fitfun(x, *p):
			A, mu, sigma, xi = p
			return A*langau_pdf(x, mu, xi, sigma)

		# xi is set equal for both since the reflect peak should have the same area as prompt?
		def double_fitfun(x, *p):
			A1, mu1, sigma1, xi1, A2, mu2, sigma2, y0 = p
			return fitfun(x, A1, mu1, sigma1, xi1) + fitfun(x, A2, mu2, sigma2, xi1) + y0

		rough_peaks_indices = find_peaks(-ydata, height=(-0.6)*ydata.min(), distance = 22)[0]
		rough_peaks = xdata[rough_peaks_indices]

		xp1Cut = (xdata > (rough_peaks[0]-30*25/256)) & (xdata < (rough_peaks[1]+25*25/256))
		xdata_cut = xdata[xp1Cut]
		ydata_cut = ydata[xp1Cut]

		param_bounds=([-1000000, 0, 0.05, 0.1, -1000000, 0.05, 0.1, -100],[-10, 256*25/256, 10, 10, -10, 256*25/256, 10, 100])
		p0 = [-np.absolute(ydata_cut).max(), 1.0*rough_peaks[0], 0.5, 0.5, -np.absolute(ydata_cut).max(), 1.0*rough_peaks[1], 0.5, 0.]

		coeffs, var_matrix2 = curve_fit(double_fitfun, xdata_cut, ydata_cut, p0=p0, bounds=param_bounds)

		peaks = np.array((coeffs[5],coeffs[1]))

		# Now doing CFD peaks
		peaks_cfd = []
		cfd_val = 0.1

		# Find minimum y-val of the prompt pulse's langaus fit
		#	Lambda function defines the prompt pulse's langaus fit
		# 	fmin finds the x value that produces a minimum in the fit, starting from mu1.
		#	fitfun returns the y-val associated with that x-val
		#	lamfun returns the prompt pulse's langaus fit shifted down by 10% of the pulse max
		#	fsolve finds the roots of the offset first pulse, effectively the x-val that gives 10% the pulse max
		#	.min() returns the lesser x-val, on the leading edge, since generally the langaus will have two that cross 10%
		funmin1 = fitfun(fmin(lambda x: fitfun(x, *coeffs[:4]), coeffs[1], disp=False), *coeffs[:4])
		lamfun1 = lambda x: fitfun(x, *coeffs[:4]) - funmin1*cfd_val
		peaks_cfd.append(fsolve(lamfun1, [coeffs[1]-10*25/256]).min())

		reflect_popt = (coeffs[4], coeffs[5], coeffs[6], coeffs[3])
		funmin2 = fitfun(fmin(lambda x: fitfun(x, *reflect_popt), coeffs[5], disp=False), *reflect_popt)
		lamfun2 = lambda x: fitfun(x, *reflect_popt) - funmin2*cfd_val
		peaks_cfd.append(fsolve(lamfun2, [coeffs[5]-10*25/256]).min())

		peaks_cfd = np.array(peaks_cfd)

		if display:
			print("Rough:",  rough_peaks.max()-rough_peaks.min())
			print("MPV:",  peaks.max()-peaks.min())
			print("CFD:",  peaks_cfd.max()-peaks_cfd.min())

			fig, (ax1) = plt.subplots(1, 1)
			ax1.plot(xdata, ydata, label="Pulse")
			ax1.plot(xdata_cut, fitfun(xdata_cut,*coeffs[:4]), label="Double fit (first peak)", color='pink')
			ax1.plot(xdata_cut, fitfun(xdata_cut,*(coeffs[4], coeffs[5], coeffs[6], coeffs[3])), label="Double fit (second peak)", color='darksalmon')
			ax1.plot(xdata_cut, double_fitfun(xdata_cut,*coeffs), label="doublefit", color='magenta')

			# Display MPV peaks
			for x in peaks:
				ax1.axvline(x, color="blue")

			# Display CFD peaks
			for x in peaks_cfd:
				ax1.axvline(x, color="black")

			ax1.text(90*25/256,-1144, f'$\Delta t =$ {round((peaks.max()-peaks.min()), 2)}', fontdict={'size': 14})

			ax1.set_xlabel("Sample times")
			ax1.set_ylabel("ADC value (ped corrected)")
			ax1.legend()
			ax1.xaxis.set_ticks_position('both')
			ax1.yaxis.set_ticks_position('both')
			fig.tight_layout()
			plt.minorticks_on()

			plt.show()

		if METHOD.lower() == 'mpv':
			return (peaks.max()-peaks.min())
		elif METHOD.lower() == 'cfd':
			return (peaks_cfd.max() - peaks_cfd.min())

	def find_l_pos_spline(self, xdata, ydata, display=False):

		# Determines the indices of the peaks in the prompt and reflected pulses
		height_cutoff = -0.6*ydata.max()
		distance_between_peaks = 20		# in units of indices
		peak_region_radius = 15			# in units of indices
		peaks_rough = find_peaks(-1*ydata, height=height_cutoff, distance=distance_between_peaks)[0]
		prompt_peak_index, reflect_peak_index = np.sort(peaks_rough[ydata[peaks_rough].argsort()[0:2]])
	
		# Creates subregions of data around the reflect peak
		reflect_lbound = reflect_peak_index - int((reflect_peak_index-prompt_peak_index)/2)-5 # lower bound is a bit left of the midway between peaks
		reflect_ubound = reflect_peak_index + 6
		ydata_subrange = ydata[reflect_lbound:reflect_ubound]
		subdomain = xdata[reflect_lbound:reflect_ubound]
		peak_region_lower, peak_region_upper = xdata[reflect_peak_index-3], xdata[reflect_peak_index+3]

		# Solves for the extrema of the reflect peak
		spline_tuple = splrep(subdomain, ydata_subrange, k=3, s=10000)
		bspline = BSpline(*spline_tuple)
		dbspline = bspline.derivative()
		reflect_cspline = CubicSpline(subdomain, bspline(subdomain))
		dcubic_spline = CubicSpline(subdomain, dbspline(subdomain))
		extrema = dcubic_spline.solve(0, extrapolate=False)
		reflect_peak_max_time = extrema[(extrema > peak_region_lower) & (extrema < peak_region_upper)][0]
		reflect_peak_max = reflect_cspline(reflect_peak_max_time)	# finds the extrema that is near our original find_peaks value
		reflect_peak_min_val = reflect_cspline(extrema[0]) + 0.1*(reflect_peak_max - reflect_cspline(extrema[0]))

		# repeating the spline for the prompt peak now
		prompt_lbound = prompt_peak_index - 14
		prompt_ubound = prompt_peak_index + 4
		prompt_subrange = ydata[prompt_lbound:prompt_ubound]
		prompt_subdomain = xdata[prompt_lbound:prompt_ubound]
		prompt_tuple = splrep(prompt_subdomain, prompt_subrange, k=3, s=10000)
		prompt_bspline = BSpline(*prompt_tuple)
		prompt_dbspline = prompt_bspline.derivative()
		prompt_cspline = CubicSpline(prompt_subdomain, prompt_bspline(prompt_subdomain))
		prompt_dcspline = CubicSpline(prompt_subdomain, prompt_dbspline(prompt_subdomain))
		prompt_extrema = prompt_dcspline.solve(0)
		peak_region_lower, peak_region_upper = xdata[prompt_peak_index-3], xdata[prompt_peak_index+3]
		prompt_peak_max_time = prompt_extrema[(prompt_extrema > peak_region_lower) & (prompt_extrema < peak_region_upper)][0]
		prompt_peak_max = prompt_cspline(prompt_peak_max_time)

		fraction_of_peak = reflect_peak_min_val/reflect_peak_max + (0.9-reflect_peak_min_val/reflect_peak_max)/2

		prompt_cfd_time = prompt_cspline.solve(fraction_of_peak*prompt_peak_max, extrapolate=False)[0]
		reflect_cfd_time = reflect_cspline.solve(fraction_of_peak*reflect_peak_max, extrapolate=False)[0]

		if display:
			fig3, ax3 = plt.subplots()
			ax3.scatter(xdata, ydata, marker='.', label='Raw data')

			reflect_peak_spline_domain = np.linspace(xdata[reflect_lbound], xdata[reflect_ubound-1], 100)
			ax3.plot(reflect_peak_spline_domain, reflect_cspline(reflect_peak_spline_domain), color='orange')

			prompt_peak_spline_domain = np.linspace(xdata[prompt_lbound], xdata[prompt_ubound-1], 100)
			ax3.plot(prompt_peak_spline_domain, prompt_cspline(prompt_peak_spline_domain), color='green')

			ax3.axhline(reflect_peak_min_val, color='pink', label=f'{round(100*reflect_peak_min_val/reflect_peak_max, 2)}% of reflected peak max')
			ax3.axvline(prompt_cfd_time, color='red')
			ax3.axvline(reflect_cfd_time, color='purple')
			ax3.legend()
			ax3.set_xlabel('Sample')
			ax3.set_ylabel('ADC Count')
			ax3.set_title('Integration bounds for the autocorrelation function')
			plt.show()

		# delta_t = reflect_peak_max_time - prompt_peak_max_time
		delta_t = reflect_cfd_time - prompt_cfd_time

		return delta_t

	def find_l_pos_cfd(self, y_data):
		"""xxx add description
		
		"""

		x_data = np.linspace(0,255,256)		

		shift = 7

		y_data_shift = np.roll(y_data, shift)
		y_data_shift[0:shift] = np.zeros(shift)

		atten = 0.1
		y_data = -1*atten*y_data

		new_data = y_data + y_data_shift

		fig, ax = plt.subplots()

		ax.plot(x_data, y_data, label='Atten. inv.')
		ax.plot(x_data, y_data_shift, label='Delayed')
		ax.plot(x_data, new_data, label='Sum')

		ax.legend()

		plt.show()

		return

	def find_t_pos_simple(self, waveform):
		return self.largest_signal_ch(waveform)

	def find_t_pos_ls_gauss(self, waveform, display=False):
		"""xxx
		
		"""

		xmin = 0
		xmax = 29
		xdata = np.linspace(xmin, xmax, xmax-xmin+1)
		ydata = (-1)*waveform.min(axis=1)

		def gauss_const_back(x, N, sigma, mu, A):
			return (N/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2) + A
		
		N_guess = 10*max(ydata)
		sigma_guess = 0.1*(xdata[-1]-xdata[0])
		mu_guess = ydata.argmax()
		A_guess = min(ydata)
		p0 = [N_guess, sigma_guess, mu_guess, A_guess]
		popt, pcov = curve_fit(gauss_const_back, xdata, ydata, p0=p0)

		if display:

			domain = np.linspace(xmin, xmax, 200)

			fig, ax = plt.subplots()
			ax.scatter(xdata, ydata, label='Raw')
			ax.plot(domain, gauss_const_back(domain, *p0), label='p0')
			ax.plot(domain, gauss_const_back(domain, *popt), label='popt')
			ax.legend()
			plt.show()

		return popt[2]

	def plot_centers(self, centers):

		mm_per_ns = 72
		offset_in_ns = 3.5
		mm_per_strip = 6.6

		fig, ax = plt.subplots()
		fig.set_size_inches([10.5,8])

		xbins, ybins = np.linspace(0,200,201), np.linspace(0,200,201)
		h, xedges, yedges, image_mesh = ax.hist2d((centers[:,0]-offset_in_ns)*mm_per_ns,centers[:,1]*mm_per_strip, bins=(xbins, ybins))#, norm=matplotlib.colors.LogNorm())
		ax.set_xlabel("dt(pulse, reflection)*v [mm]")
		ax.set_ylabel("Y position (perpendicular to strips) [mm]")
		fig.colorbar(image_mesh, ax=ax)

		fig2, ax2 = plt.subplots()
		y_data = h[np.max(h, axis=1).argmax()]
		x_data = xedges[:-1]
		ax2.scatter(x_data, y_data)

		# this fit thing isn't working yet
		# def gauss_func(x, N, sigma, mu, A):
		# 	return (N/np.sqrt(2*np.pi)*sigma)*np.exp(-1*(x-mu)*(x-mu)/(2*sigma*sigma)) + A
		# N_guess = 10*max(y_data)
		# sigma_guess = 0.1*(x_data[-1]-x_data[0])
		# mu_guess = 0.5*(x_data[-1]+x_data[0])
		# A_guess = min(y_data)
		# p0 = [N_guess, sigma_guess, mu_guess, A_guess]
		# popt, pcov = curve_fit(gauss_func, x_data, y_data, p0=p0)
		# x_data_domain = np.linspace(x_data[0], x_data[-1], 200)
		# ax2.plot(x_data_domain, gauss_func(x_data_domain, *popt))

		plt.show()

		return

	#the calibration file is an .h5 file that holds a pandas dataframe
	#that has the same dataframe structure as is listed above for self.df. 
	#The one difference for simplicity is that the self.sync_dict is contained
	#within this dataframe and is identified by a "sync" flag of 0 or 1
	def load_calibration(self, clear=False):
		if(clear):
			self.initialize_dataframe()

		if(self.calibration_fn is None):
			print("No configuration file selected on initializing Acdc objects, using default values")
			chs = range(30)
			strip_space = 6.9 #mm
			for ch in chs:
				if(ch == self.sync_ch):
					self.sync_dict["times"] = np.append(np.linspace(0, 255*self.dt, 255), 500) #picoseconds, timebase for each sample, 500ps is the wraparound time
					continue 
				
				#edit the entries of the channel
				self.df.at[ch, "waveform"] = None #load in on event loop 
				self.df.at[ch, "position"] = strip_space*ch
				self.df.at[ch, "len_cor"] = 0
				self.df.at[ch, "time_offsets"] = np.append(np.linspace(0, 255*self.dt, 255), 500) #picoseconds, timebase for each sample, 500ps is the wraparound time
				self.df.at[ch, "voltage_count_curves"] = [0,0]*256

		#otherwise, if a calibration file is included, use it
		else:
			c = pd.read_hdf(self.calibration_fn) #this is an .h5 file that contains an empty dataframe but with calibration parameters
			#this will check that the columns in the calibration
			#file are also the columns of the present class definitions DF
			check_cols = self.columns + ["sync"]
			if(c.columns != check_cols):
				print("Columns in calibration file are: ", end='')
				print(c.columns)
				print("Was expecting: ",end='')
				print(check_cols)
				print("Loading calibration may fail... trying anyway")

			#handle the synchronization channel first
			sync_row = c[c['sync'] == 1] #selects only the row where sync == 1
			self.sync_dict["ch"] = sync_row["ch"]
			self.sync_dict["times"] = sync_row["times"]
			self.sync_dict["wraparound"] = sync_row["wraparound"]

			#grab the channels that are not sync
			ch_df = c[c['sync'] != 1]
			#drop the "sync" column entirely
			ch_df = ch_df.drop("sync", axis=1)
			#now this calibration dict is identical to a "waveform empty" self.df
			self.df = ch_df 

		print("Calibration loaded for ACDC {:d} in LAPPD station {:d} with LAPPD ID {:d}".format(self.id, self.lappd_station, self.lappd_id))


	#a function used by the datafile parser to update the ACDC class on an event by event basis
	#without re-updating all of the globally constant calibration data.
	#"waves" is a dictionary of np.arrays like waves[ch] = np.array(waveform samples)
	def update_waveforms(self, waves, timestamps_320, timestamps):
		#a dictionary of waveforms, with channel numbers, includes the sync wave
		waves = np.array(waves)
		self.cur_times_320 = timestamps_320
		self.cur_times = timestamps
		self.cur_event_count = waves.shape[0]
		for ch in waves.shape[1]:
			if(ch == self.sync_dict["ch"]):
				self.sync_dict["waveform"] = waves[:, ch]#TODO: check this indexing works
				continue 
			self.df.at[ch, "waveform"] = waves[:, ch]
			

	#Correcting the raw waveform #1!
	#Jin suggests the ACDC class should carry a chain of waveforms rather then every correction function directly acting on a single waveform variable; the former is much more traceable and debuggable.
	def baseline_subtract(self):
		pass#NOT YET IMPLEMENTED

	#Correcting the raw waveform #2!
	def voltage_linearization(self):
		pass#NOT YET IMPLEMENTED

	def check_coincidence_assign_event_numbers(acdcs, window=30):
		for a in acdcs:
			for e in a.cur_event_count:
				a.event_numbers.append(e)#Right now, first event in the dataframe is event 0, second is event 1, etc. TODO: assign event numbers appropriately after coincidence check.

		#do math to look at coincidence of clocks. 
		return 0 #or 1, or a list of those that are in coincidence vs those that are not.


if __name__=='__main__':

	# initialization dict
	init_dict = {
		'acdc_id': 1,
		'lappd_id': 1,
		'station_id': 1,
		'sync_ch': 0,
		'strip_pos': None,
		'len_cor': None,
		'times': None,			 # xxx need a better name for this
		'wraparound': None,
		'vel': 0.18,			 # mm/ps, average (~500 MHz - 1GHz) propagation velocity of the strip 
		'dt': 1.0/(40e6*256),	 # picoseconds, nominal sampling time interval, 1/(clock to PSEC4 x number of samples)
		# 'pedestal_data_path': r'/home/cameronpoe/Desktop/lappd_tof_container/testData/old_data/Raw_testData_20230615_164912_b0.txt',
		'pedestal_data_path': r'/home/cameronpoe/Desktop/lappd_tof_container/testData/ped_Raw_testData_ACC1_20230714_093238_b0.txt',
		'pedestal_counts': None,
		'pedestal_voltage': None,
		'voltage_count_curves': None,
		'calib_data_file_path': r'/home/cameronpoe/Desktop/lappd_tof_container/LAPPD_TOF_Analysis/new_new_test.root'
	}

	# data_path = r'/home/cameronpoe/Desktop/lappd_tof_container/testData/old_data/Raw_testData_20230615_170611_b0.txt'
	data_path = r'/home/cameronpoe/Desktop/lappd_tof_container/testData/Raw_testData_ACC1_20230714_094508_b0.txt'

	test_acdc = Acdc(init_dict)

	# test_acdc.import_raw_data(data_path)

	# file_name_i_want_to_save_as = r'current_working_data'
	# directory_to_save_to = r'/home/cameronpoe/Desktop/lappd_tof_container/testData/processed_data'
	# test_acdc.save_data_npz(file_name_i_want_to_save_as, directory_path=directory_to_save_to)
	
	test_acdc.load_data_npz(r'testData/processed_data/current_working_data.npz')
	# test_acdc.hist_single_cap_counts_vs_ped(10, 22)

	# test_acdc.plot_ped_corrected_pulse(154)

	# test_acdc.plot_raw_lappd(350)

	event_subset = np.linspace(500, 3000, 2501, dtype=int)
	# bad_events = [554, 592, 593, 594, 632, 636, 709, 783, 854, 878, 887, 923, 962, 1033, 1047, 1099, 1139, 1180, 1240]
	# bad_events = [616, 714, 1074, 1162, 1174]
	# bad_events = [783]
	centers = test_acdc.find_event_centers(METHOD='langaus_mpv', DEBUG_EVENTS=True, events=560)
	
	exit()


	single_time = 500
	single_channel = 15
	y_data = waveform[single_time, single_channel, :]
	x_data = np.linspace(0, len(y_data), num=len(y_data))

	fig, ax = plt.subplots()
	ax.set_xlabel('Capacitor number')
	ax.set_ylabel('ADC Counts')
	ax.plot(x_data, y_data)
	plt.show()




