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
import pylandau

avg_fwhms = []

# Some quick helper functions

def convert_to_list(some_object):
	"""Converts an object into a single-element list if the object is not already a list. Helps when a function works by going through elements of a list, but you want to pass a single element to that function.
	Arguments:
		(any) some_object: what you want to convert to a list (if not already a list)
	"""

	if not isinstance(some_object, list) and not isinstance(some_object, np.ndarray):
		some_object = [some_object]

	return some_object


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

		# have to normalize since we get voltage as a value on [0,4096]
		voltage_counts[:,:,:,0] = voltage_counts[:,:,:,0]*1.2/4096.
		
		def lineraize_wrap(f, val):
			try:
				return f(val)
			except(ValueError):
				if val < 2000:
					return 0
				else:
					return 3.3

		
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

	def find_event_centers(self, events=None):
		"""xxx add description

		"""

		# If no `events` list is passed, assume we must find centers for all events in cur_waveforms. 
		if events is None:
			events = np.linspace(0, len(self.sample_times)-1, len(self.sample_times), dtype=int)
		else:
			events = convert_to_list(events)

		centers = []

		num_skipped_waveforms = 0
		for i in events:

			if i%500 == 0:
				print(f'{i} centers calculated...')

			try:

				waveform = self.cur_waveforms[i]	
				
				l_pos_x_data = self.sample_times[i]
				largest_ch = self.largest_signal_ch(waveform)
				l_pos_y_data = waveform[largest_ch]
				

				DISPLAY_CENTER_FITS = True
				DIAGNOSTIC_DATA = True
				if DISPLAY_CENTER_FITS:
					self.plot_ped_corrected_pulse(i, channels=largest_ch)
				l_pos = self.find_l_pos_autocor_le_subset(l_pos_x_data, l_pos_y_data, display=DISPLAY_CENTER_FITS, diagnostic=DIAGNOSTIC_DATA)
				# l_pos = self.find_l_pos_cfd(l_pos_y_data)
				# l_pos = self.find_l_pos_autocor_centered(l_pos_y_data)
				# l_pos = self.find_l_pos_spline(l_pos_y_data)
				# l_pos = self.find_l_pos_langaus(l_pos_y_data)
				# l_pos = 1.2

				t_pos = self.find_t_pos_ls_gauss(waveform, display=DISPLAY_CENTER_FITS)
				# t_pos = self.find_t_pos_simple(waveform)
				# t_pos = 1.2

				centers.append((l_pos, t_pos))

			except:
				print(f'Error with event {i}')
				num_skipped_waveforms += 1
				pass

		
		print(f'Number of skipped waveforms: {num_skipped_waveforms}')
		print(f'Total number of waveforms: {len(events)}')
		print(f'Percent skipped: {round(100*num_skipped_waveforms/len(events), 2)}%')
						
		centers = np.array(centers)

		np.save('centers_le_spec_autocor', centers)		

		selection = centers[:,0] > 16
		centers[:,0][selection] = 25 - centers[:,0][selection]

		return centers
	
	def largest_signal_ch(self, waveform):
		"""Small helper function to retrieve the channel to be used in the find_l_pos functions.
		Arguments:
			(Acdc) self
			(np.array) waveform: a single event's waveform (2D array) from which the optimal channel will be found		
		"""

		return np.absolute(waveform).max(axis=1).argmax()
	
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

	def find_l_pos_spline(self, ydata):
		"""Finds the longitudinal position (l_pos) of the incident particle using a spline fitting method.
		xxx add more
		
		"""

		xdata = np.linspace(0, 255, 256)
		ydata = -1*ydata

		height_cutoff = 0.5*ydata.max()
		distance_between_peaks = 20
		peak_region_radius = 15

		peaks_rough = scipy.signal.find_peaks(ydata, height=height_cutoff, distance=distance_between_peaks)[0]
		peaks_precise = []

		for rough_peak in peaks_rough:

			peak_region_cut = (xdata > (rough_peak-peak_region_radius)) & (xdata < (rough_peak+peak_region_radius))

			spline_tuple = splrep(xdata[peak_region_cut], ydata[peak_region_cut], k=3, s=10000)
			data_bspline = BSpline(*spline_tuple)
			ddata_bspline = data_bspline.derivative()
			
			peak_region_domain = np.linspace(rough_peak-peak_region_radius, rough_peak+peak_region_radius, 100)
			dcubic_spline = CubicSpline(peak_region_domain, ddata_bspline(peak_region_domain))

			extrema = dcubic_spline.solve(0)

			extrema = extrema[(extrema > (rough_peak-12)) & (extrema < (rough_peak+12))]

			if len(extrema) > 0:
				extrema = extrema[data_bspline(extrema).argsort()][-1]
			else:
				extrema = rough_peak
			
			peaks_precise.append(extrema)
		

		peaks_precise = np.array(peaks_precise)

		# Thinking this needs to change (more precise)
		peaks_precise = peaks_precise*25./256
		l_pos = peaks_precise.max()-peaks_precise.min()
		
		return l_pos

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

		integrals = []
		lags = []

		lag = 0
		lag_increment = 1
		ydata_shifted = np.copy(ydata)
		xdata_shifted = np.copy(xdata)
		while lag < 256:

			xdata_shifted -= (25/256)*lag_increment*np.ones_like(xdata_shifted)
			indices_inbounds = np.linspace(0,255,256,dtype=int)[(xdata_shifted > integral_lower_bound) & (xdata_shifted < integral_upper_bound)]
			if len(indices_inbounds) == 0:
				lag += lag_increment
				continue

			xdata_shifted_inbounds = xdata_shifted[indices_inbounds]
			
			ydata_inbounds = ydata_shifted[indices_inbounds]
			ydata_stationary = prompt_cspline(xdata_shifted_inbounds)

			# standard autocor method
			# single_integral = np.sum(ydata_inbounds*ydata_stationary)
			
			# trapezoidal integral method
			left_outbounds_ind = indices_inbounds[0] - 1
			if not left_outbounds_ind < 0:
				left_slope = (ydata_inbounds[0] - ydata_shifted[left_outbounds_ind])/(xdata_shifted_inbounds[0] - xdata_shifted[left_outbounds_ind])
				left_border_yval = ydata_inbounds[0] + left_slope*(integral_lower_bound-xdata_shifted_inbounds[0])
				xdata_shifted_inbounds = np.insert(xdata_shifted_inbounds, 0, integral_lower_bound)
				ydata_inbounds = np.insert(ydata_inbounds, 0, left_border_yval)

			right_outbounds_ind = indices_inbounds[-1] + 1
			if right_outbounds_ind != 256:
				right_slope = (ydata_shifted[right_outbounds_ind]-ydata_inbounds[-1])/(xdata_shifted[right_outbounds_ind]-xdata_shifted_inbounds[-1])
				right_border_yval = ydata_inbounds[-1] + right_slope*(integral_upper_bound-xdata_shifted_inbounds[-1])
				xdata_shifted_inbounds = np.append(xdata_shifted_inbounds, integral_upper_bound)
				ydata_inbounds = np.append(ydata_inbounds, right_border_yval)
								
			ydata_stationary = prompt_cspline(xdata_shifted_inbounds)

			single_integral = trapezoid(ydata_inbounds*ydata_stationary, xdata_shifted_inbounds)

			# chi-squared method
			# single_integral = np.sum((ydata_inbounds - ydata_stationary)*(ydata_inbounds - ydata_stationary))

			integrals.append(single_integral)
			lags.append((25/256)*lag)

			lag += lag_increment

		integrals = np.array(integrals)
		lags = np.array(lags)

		height_cutoff = 0.6*integrals.max()
		distance_between_peaks = 5				# in units of indices
		peak_region_radius = 5*(25/256)			# in units of ns

		if diagnostic:
			large_region_radius = 15*(25/256)		# in units of ns
			if diagnostic_method == 'gauss':
				def diag_func(x, N, sigma, mu):
						return (N/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(x-mu)*(x-mu)/(sigma*sigma))
			elif diagnostic_method == 'langau':
				def diag_func(x, A, mu, xi, sigma):
					return A*pylandau.langau_pdf(x, mu, xi, sigma)
			diag_funcs = []

		integral_peaks_rough_indices = find_peaks(integrals, height=height_cutoff, distance=distance_between_peaks)[0]
		integral_peaks_rough_times = lags[integral_peaks_rough_indices]

		extremas = []
		splines = []
		domains = []
		for integral_peak_rough in integral_peaks_rough_times:

			integral_peak_region_cut = (lags > (integral_peak_rough-peak_region_radius)) & (lags < (integral_peak_rough+peak_region_radius))

			lags_cut = lags[integral_peak_region_cut]
			integrals_cut = integrals[integral_peak_region_cut]

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

				integrals_cut_diag = integrals[integral_peak_region_cut_diag]
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
			ax2.scatter(lags, integrals, label='Discrete Autocorrelation', marker='.')
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
				ax.scatter(lags, integrals, label='Discrete Autocorrelation', marker='.')
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

	def find_l_pos_autocor_centered(self, ydata):
		"""xxx fill in description
		
		"""

		# fig, ax = plt.subplots()

		domain = np.linspace(0,255,256)


		height_cutoff = -0.6*ydata.max()
		distance_between_peaks = 20
		peak_region_radius = 15

		peaks_rough = scipy.signal.find_peaks(-1*ydata, height=height_cutoff, distance=distance_between_peaks)[0]
		
		left_peak, right_peak = np.sort(peaks_rough[ydata[peaks_rough].argsort()[0:2]])

		integrals = []
		lags = []

		lag = 0
		lag_increment = 3
		while lag < 256:

			ydata_shifted = np.copy(ydata)	# make sure not to change ydata

			if lag != 0:
				ydata_shifted[-lag:] = 0

			ydata_shifted = np.roll(ydata_shifted, lag)
			auto_cor_func = ydata*ydata_shifted
			# ax.plot(domain, auto_cor_func)

			subdomain = np.linspace(right_peak-peak_region_radius, right_peak+peak_region_radius, 2*peak_region_radius+1, dtype=int)

			integrals.append(scipy.integrate.trapezoid(auto_cor_func[subdomain], domain[subdomain]))
			lags.append(lag)

			lag += lag_increment

		integrals = np.array(integrals)
		lags = np.array(lags)
		integral_peaks_rough = lags[scipy.signal.find_peaks(integrals, height=0.6*(integrals.max()), distance=20/lag_increment)[0]]

		integral_peak_rough = integral_peaks_rough[0]

		integral_peak_region_cut = (lags > (integral_peak_rough-peak_region_radius)) & (lags < (integral_peak_rough+peak_region_radius))

		spline_tuple = splrep(lags[integral_peak_region_cut], integrals[integral_peak_region_cut], k=3, s=10000)
		data_bspline = BSpline(*spline_tuple)
		ddata_bspline = data_bspline.derivative()
		
		peak_region_domain = np.linspace(integral_peak_rough-peak_region_radius, integral_peak_rough+peak_region_radius, 100)
		dcubic_spline = CubicSpline(peak_region_domain, ddata_bspline(peak_region_domain))

		extrema = dcubic_spline.solve(0)

		extrema = extrema[(extrema > (integral_peak_rough-peak_region_radius+3)) & (extrema < (integral_peak_rough+peak_region_radius-3))]

		if len(extrema) > 0:
			extrema = extrema[data_bspline(extrema).argsort()][-1]
		else:
			extrema = integral_peak_rough

		fig2, ax2 = plt.subplots()
		# ax2.plot(lags, integrals)
		ax2.scatter(lags*25./256, integrals, label='Raw Data')
		ax2.plot(peak_region_domain*25./256, data_bspline(peak_region_domain), label='Spline Fit', color='orange')
		ax2.axvline(extrema*25./256, color='red', label=f'Extrema: {round(extrema*25./256, 3)} ns')
		ax2.legend()
		ax2.set_xlabel('Time delay (ns)')
		ax2.set_ylabel('Integral value')

		plt.show()
		
		return (25./256)*extrema

	def find_l_pos_autocor(self, ydata, ver='LE_DELTA', display=True):
		"""xxx fill in description
		
		"""

		ver = 'LE_DELTA' # can be FULL, CENTERED, LE_PEAK, LE_DELTA, or LE_SPEC_BOUND
		display=True

		domain = np.linspace(0,255,256, dtype=int)
		# fig3, ax3 = plt.subplots()
		# ax3.plot(domain, ydata)

		# ydata = np.roll(ydata, cw_low)
		# ax3.plot(domain, ydata)
		# domain = np.roll(domain, cw_low)
		# ax3.plot(domain, ydata)

		# plt.show()

		height_cutoff = -0.6*ydata.max()
		distance_between_peaks = 20
		peak_region_radius = 15
		peaks_rough = scipy.signal.find_peaks(-1*ydata, height=height_cutoff, distance=distance_between_peaks)[0]
		prompt_peak, reflect_peak = np.sort(peaks_rough[ydata[peaks_rough].argsort()[0:2]])

		if ver == 'FULL':
			integral_lower_bound = domain[0]
			integral_upper_bound = domain[-1]
		elif ver == 'CENTERED':
			integral_lower_bound = prompt_peak-10
			integral_upper_bound = prompt_peak+10

			if display:
				fig, ax = plt.subplots()
				ax.scatter(domain[integral_lower_bound-5:integral_upper_bound+5], ydata[domain[integral_lower_bound-5:integral_upper_bound+5]], marker='.')
				# ax.plot(np.linspace(fit_lower_bound, fit_upper_bound, 100), cubic_spline(np.linspace(fit_lower_bound, fit_upper_bound, 100)), color='orange')
				# ax.axhline(0.1*prompt_peak_max, color='purple')
				# ax.axhline(0.9*prompt_peak_max, color='purple')
				ax.axvline(integral_lower_bound, color='red', label='Lower bound (10%)')
				ax.axvline(integral_upper_bound, color='purple', label='Upper bound (90%)')
				ax.legend()
				ax.set_xlabel('Sample')
				ax.set_ylabel('ADC Count')
				ax.set_title('Integration bounds for the autocorrelation function')

				plt.show()

		elif 'LE' in ver:
			

			if ver == 'LE_SPEC_BOUND':
				fit_lower_bound = reflect_peak - 14
				fit_upper_bound = reflect_peak + 4
			else:
				# subtracting 12 since 12 is approximately 1.2 ns * (256/25.5). 1.2 ns is used as the rough maximum rise time 
				#	a signal could have.
				fit_lower_bound = prompt_peak - 14

				# add a small amount to rightside of peak for the spline fit
				fit_upper_bound = prompt_peak + 4

			ydata_subrange = ydata[fit_lower_bound:fit_upper_bound]
			subdomain = domain[fit_lower_bound:fit_upper_bound]

			spline_tuple = splrep(subdomain, ydata_subrange, k=3, s=10000)
			bspline = BSpline(*spline_tuple)
			dbspline = bspline.derivative()
			cubic_spline = CubicSpline(subdomain, bspline(subdomain))
			dcubic_spline = CubicSpline(subdomain, dbspline(subdomain))

			extrema = dcubic_spline.solve(0)

			if ver == 'LE_SPEC_BOUND':

				old_cubic_spline = cubic_spline
				
				reflect_peak_max = cubic_spline(extrema[(extrema > (reflect_peak-3)) & (extrema < (reflect_peak+3))])
				reflect_peak_min_val = cubic_spline(extrema[0]+3) # adding two just to not be right at the inflection point

				# repeating the spline for the prompt peak now
				fit_lower_bound = prompt_peak - 14
				fit_upper_bound = prompt_peak + 4

				ydata_subrange = ydata[fit_lower_bound:fit_upper_bound]
				subdomain = domain[fit_lower_bound:fit_upper_bound]

				spline_tuple = splrep(subdomain, ydata_subrange, k=3, s=10000)
				bspline = BSpline(*spline_tuple)
				dbspline = bspline.derivative()
				cubic_spline = CubicSpline(subdomain, bspline(subdomain))
				dcubic_spline = CubicSpline(subdomain, dbspline(subdomain))

				extrema = dcubic_spline.solve(0)

				prompt_peak_max = cubic_spline(extrema[(extrema > (prompt_peak-3)) & (extrema < (prompt_peak+3))])

				integral_lower_bound = cubic_spline.solve(reflect_peak_min_val, extrapolate=False)[0]
				integral_upper_bound = cubic_spline.solve(0.9*prompt_peak_max, extrapolate=False)[0]

				reflect_peak_max = reflect_peak_max[0]
				prompt_peak_max = prompt_peak_max[0]

				if display:
					fig3, ax3 = plt.subplots()
					ax3.scatter(domain[prompt_peak-19:reflect_peak+9], ydata[domain[prompt_peak-19:reflect_peak+9]], marker='.')
					ax3.plot(np.linspace(reflect_peak-14, reflect_peak+4, 100), old_cubic_spline(np.linspace(reflect_peak-14, reflect_peak+4, 100)), color='orange')
					ax3.plot(np.linspace(prompt_peak-14, prompt_peak+4, 100), cubic_spline(np.linspace(prompt_peak-14, prompt_peak+4, 100)), color='green')
					ax3.axhline(reflect_peak_min_val, color='pink', label=f'{round(100*reflect_peak_min_val/reflect_peak_max, 2)}% of reflected peak max')
					ax3.axvline(integral_lower_bound, color='red', label=f'Lower bound ({round(100*reflect_peak_min_val/prompt_peak_max, 2)}% of \nprompt peak max)')
					ax3.axvline(integral_upper_bound, color='purple', label='Upper bound (90% of \nprompt peak max)')
					ax3.legend()
					ax3.set_xlabel('Sample')
					ax3.set_ylabel('ADC Count')
					ax3.set_title('Integration bounds for the autocorrelation function')

			else:
				prompt_peak_max = cubic_spline(extrema[(extrema > (prompt_peak-3)) & (extrema < (prompt_peak+3))])

				integral_lower_bound = cubic_spline.solve(0.1*prompt_peak_max, extrapolate=False)[0]
				# integral_lower_bound = integral_lower_bound[]
				integral_upper_bound = cubic_spline.solve(0.9*prompt_peak_max, extrapolate=False)[0]

				if display:
					fig, ax = plt.subplots()
					ax.scatter(domain[fit_lower_bound-5:reflect_peak+9], ydata[domain[fit_lower_bound-5:reflect_peak+9]], marker='.')
					ax.plot(np.linspace(fit_lower_bound, fit_upper_bound, 100), cubic_spline(np.linspace(fit_lower_bound, fit_upper_bound, 100)), color='orange')
					# ax.axhline(0.1*prompt_peak_max, color='purple')
					# ax.axhline(0.9*prompt_peak_max, color='purple')
					ax.axvline(integral_lower_bound, color='red', label='Lower bound (10%)')
					ax.axvline(integral_upper_bound, color='purple', label='Upper bound (90%)')
					ax.legend()
					ax.set_xlabel('Sample')
					ax.set_ylabel('ADC Count')
					ax.set_title('Integration bounds for the autocorrelation function')

		integrals = []
		lags = []

		lag = 0
		lag_increment = 3
		while lag < 256:

			ydata_shifted = np.copy(ydata)	# make sure not to change ydata

			if lag != 0:
				ydata_shifted[:lag] = 0

			ydata_shifted = np.roll(ydata_shifted, -1*lag)
			auto_cor_func = ydata*ydata_shifted

			subdomain = np.linspace(int(integral_lower_bound), int(integral_upper_bound), int(integral_upper_bound)-int(integral_lower_bound)+1, dtype=int)

			integrals.append(scipy.integrate.trapezoid(auto_cor_func[subdomain], domain[subdomain]))
			lags.append(lag)

			lag += lag_increment

		integrals = np.array(integrals)
		lags = np.array(lags)

		if ver != 'FULL':
			
			integral_peaks_rough = lags[scipy.signal.find_peaks(integrals, height=0.6*(integrals.max()), distance=5/lag_increment)[0]]

			if ver == 'LE_PEAK' or ver == 'CENTERED':
				integral_peaks_rough = np.array([integral_peaks_rough[-1]])
			
			extremas = []
			splines = []
			domains = []
			for integral_peak_rough in integral_peaks_rough:

				integral_peak_region_cut = (lags > (integral_peak_rough-peak_region_radius)) & (lags < (integral_peak_rough+peak_region_radius))

				spline_tuple = splrep(lags[integral_peak_region_cut], integrals[integral_peak_region_cut], k=3, s=10000)
				data_bspline = BSpline(*spline_tuple)
				ddata_bspline = data_bspline.derivative()
				
				peak_region_domain_lower = integral_peak_rough-peak_region_radius
				if peak_region_domain_lower < 0:
					peak_region_domain_lower = 0


				peak_region_domain = np.linspace(peak_region_domain_lower, integral_peak_rough+peak_region_radius, 100)
				dcubic_spline = CubicSpline(peak_region_domain, ddata_bspline(peak_region_domain))

				extrema = dcubic_spline.solve(0)

				extrema = extrema[(extrema > (integral_peak_rough-peak_region_radius+3)) & (extrema < (integral_peak_rough+peak_region_radius-3))]

				if len(extrema) > 0:
					extrema = extrema[data_bspline(extrema).argsort()][-1]
				else:
					extrema = integral_peak_rough
				
				extremas.append(extrema)
				splines.append(data_bspline)
				domains.append(peak_region_domain)

		extremas = np.array(extremas)

		if display:
			fig2, ax2 = plt.subplots()
			# ax2.plot(lags, integrals)
			ax2.scatter(lags*25./256, integrals, label='Raw Data')
			if ver != 'FULL':
				for i, extrema in enumerate(extremas):
					if i == 0:
						color1 = 'orange'
						color2 = 'red'
					elif i == 1:
						color1 = '#66ff00'
						color2 = 'pink'

					ax2.plot(domains[i]*25./256, splines[i](domains[i]), label='Spline Fit', color=color1)
					ax2.axvline(extrema*25./256, label=f'Extrema: {round(extrema*25./256, 3)} ns', color=color2)
			ax2.legend()
			ax2.set_xlabel('Time delay (ns)')
			ax2.set_ylabel('Integral value')
			if ver == 'LE_DELTA' or ver == 'LE_SPEC_BOUND':
				ax2.text(16.5, 3.33e6, f'$\Delta t=${round((25./256)*(extremas[1]-extremas[0]), 3)} ns', fontdict={'size': 16})

			plt.show()
		
		if ver == 'LE_DELTA' or ver == 'LE_SPEC_BOUND':
			delta_t = (25./256)*(extremas[1] - extremas[0])
		else:
			delta_t = (25./256)*extrema
		return delta_t

	def find_l_pos_langaus(self, ydata):

		display=False

		def fitfun(x, *p):
			A, mu, sigma, xi = p
			#return A*np.exp(-(x-mu)**2/(2.*sigma**2))
			#return A*landau.pdf(x=x,x_mpv=mu,xi=sigma)
			#return A*langauss.pdf(x=x,landau_x_mpv=mu,landau_xi=xi,gauss_sigma=sigma)
			return A*pylandau.langau_pdf(x, mu, xi, sigma)

		def double_fitfun(x, *p):
			A1, mu1, sigma1, xi1, A2, mu2, sigma2, y0 = p
			#return A*np.exp(-(x-mu)**2/(2.*sigma**2))
			#return A*landau.pdf(x=x,x_mpv=mu,xi=sigma)
			#return A*langauss.pdf(x=x,landau_x_mpv=mu,landau_xi=xi,gauss_sigma=sigma)
			return fitfun(x, A1, mu1, sigma1, xi1) + fitfun(x, A2, mu2, sigma2, xi1) + y0


		xdata = np.linspace(0,255,256)

		peaks2 = scipy.signal.find_peaks(-ydata, height=(-0.6)*ydata.min(), distance = 22)[0]
		peaks2 = xdata[peaks2]

		peaks = []
		csl = []
		coefs = []
		# print("rough peaks", peaks2)

		xp1Cut = (xdata > (peaks2[0]-30)) & (xdata < (peaks2[1]+25))
		xdata_cut = xdata[xp1Cut]
		ydata_cut = ydata[xp1Cut]
		param_bounds=([-1000000,0, 0.1, 0.1, -1000000, 0, 0.1, -100],[-10, 256, 10, 10, -10, 256, 10, 100])
		p0 = [-1000., 1.0*peaks2[0], 5., 5., -1000., 1.0*peaks2[1], 5., 0.]
		coeff2, var_matrix2 = curve_fit(double_fitfun, xdata_cut, ydata_cut, p0=p0, bounds=param_bounds)
		#coeff2 = coefs
		# print("coefficients", coeff2)

		peaks = np.array((coeff2[5],coeff2[1]))

		try:
			# print(peaks.max()-peaks.min())
			# print(peaks2.max()-peaks2.min())
			pass
		except:
			pass

		# print(peaks2)
		# #print(peaks)
		# print((coeff2[1], coeff2[5]))
		# print(coeff2[5] - coeff2[1])

		peaks3 = []

		funmin1 = fitfun(fmin(lambda x: fitfun(x, *coeff2[:4]), coeff2[1], disp=False), *coeff2[:4])
		#print("Fmin:", funmin1)
		lamfun1 = lambda x: fitfun(x, *coeff2[:4]) - funmin1*0.1
		peaks3.append(fsolve(lamfun1, [coeff2[1]-10]).min())

		funmin2 = fitfun(fmin(lambda x: fitfun(x, *(coeff2[4], coeff2[5], coeff2[6], coeff2[3])), coeff2[5], disp=False), 
											*(coeff2[4], coeff2[5], coeff2[6], coeff2[3]))
		#print("Fmin:", funmin2)
		lamfun2 = lambda x: fitfun(x, *(coeff2[4], coeff2[5], coeff2[6], coeff2[3])) - funmin2*0.1
		peaks3.append(fsolve(lamfun2, [coeff2[5]-10]).min())

		peaks3 = np.array(peaks3)

		if display:
			print(peaks3)
			print("Rough:",  peaks2.max()-peaks2.min())
			print("MPV:",  (25./256)*(peaks.max()-peaks.min()))
			print("CFD:",  peaks3.max()-peaks3.min())

		if display:

			fig, (ax1) = plt.subplots(1, 1)
			ax1.plot(xdata, ydata, label="Pulse")
			for i, (x, bs) in enumerate(csl):
				ax1.plot(x, bs, label="SingleFit_%i"%(i+1), color='orange')
			ax1.plot(xdata_cut, fitfun(xdata_cut,*coeff2[:4]), label="doublefit_1", color='pink')
			ax1.plot(xdata_cut, fitfun(xdata_cut,*(coeff2[4], coeff2[5], coeff2[6], coeff2[3])), label="doublefit_2", color='darksalmon')
			ax1.plot(xdata_cut, double_fitfun(xdata_cut,*coeff2), label="doublefit", color='magenta')
			#ax1.plot(xdata_cut, lamfun1(xdata_cut), label="doublefit", color='black')
			#ax1.plot(xdata_cut, lamfun2(xdata_cut), label="doublefit", color='black')
			#ax1.plot(xplot, dbs(xplot))
			#ax1.plot(xplot, dcs(xplot))
			#for x in peaks:
			#    ax1.axvline(x, color="green")

			#for x in peaks2:
			#    ax1.axvline(x, color="red")
			ax1.axvline(coeff2[1], color="blue")
			ax1.axvline(coeff2[5], color="blue")
			for x in peaks3:
				ax1.axvline(x, color="black")

			ax1.set_xlabel("Sample")
			ax1.set_ylabel("ADC value (ped corrected)")

			#ax1.axvline(cw_low, color="green")
			ax1.axvline(40, color="green")

			ax1.legend()

			ax1.text(90,-1144, f'$\Delta t =$ {round((25./256)*(peaks.max()-peaks.min()), 2)}', fontdict={'size': 14})

			fig.tight_layout()

		return (25./256)*(peaks.max()-peaks.min())

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
	
	
	def butter_pass(self, cutoff, sampling_rate, bytpe, order=5):
		nyq = 0.5 * sampling_rate
		normal_cutoff = cutoff / nyq
		sos = scipy.signal.butter(N=order, Wn=cutoff, fs=sampling_rate, btype=bytpe, analog=False, output='sos')
		return sos

	def high_pass_filt(self, data, cutoff, sampling_rate, order=2):

		sos = self.butter_pass(cutoff, sampling_rate, 'hp', order=order)
		y = scipy.signal.sosfilt(sos, data)
		
		return y
	
	def low_pass_filt(self, data, cutoff, sampling_rate, order=2):

		sos = self.butter_pass(cutoff, sampling_rate, 'lp', order=order)
		y = scipy.signal.sosfilt(sos, data)
		
		return y

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
		'calib_data_file_path': r'/home/cameronpoe/Desktop/lappd_tof_container/testData/test.root'
	}

	# data_path = r'/home/cameronpoe/Desktop/lappd_tof_container/testData/old_data/Raw_testData_20230615_170611_b0.txt'
	data_path = r'/home/cameronpoe/Desktop/lappd_tof_container/testData/Raw_testData_ACC1_20230714_094508_b0.txt'

	test_acdc = Acdc(init_dict)

	test_acdc.import_raw_data(data_path)

	# test_acdc.hist_single_cap_counts_vs_ped(10, 22)

	# test_acdc.plot_ped_corrected_pulse(154)

	# test_acdc.plot_raw_lappd(350)

	# time_domain = np.linspace(0, 0.0000255, 256)

	centers = test_acdc.find_event_centers(events=938)
	print(f'Average FWHM (all events): {sum(avg_fwhms)/len(avg_fwhms)} ns')
	# centers = test_acdc.find_event_centers(events=4)
	# centers = np.load(r'/home/cameronpoe/Desktop/lappd_tof_container/LAPPD_TOF_Analysis/centers_le_spec_autocor.npy')
	# test_acdc.plot_centers(centers)
	
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




