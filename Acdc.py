import numpy as np 
import bitstruct.c as bitstruct
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline, CubicSpline, PPoly
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid
import uproot
from time import process_time, time
from multiprocessing import Pool
import numba
import warnings
import yaml


MAX_PROCESSES = 1
CALIB_ADC = True			# Toggles whether VCCs are used (true) or simple pedestal subtraction (false)
CALIB_TIME_BASE = True		# Toggles whether ellipse fit time base is used (true) or not (false)
NO_POSITIONS = True		# Toggles whether x- and y-positions are reconstructed
NO_SINES = True				# Toggles whether sync channel sines are fitted
EXCLUDE_WRAP = True			# Toggles whether wraparound is excluded from sync sine fit
VAR_SINE_FREQ = False		# Toggles whether sync sine fit frequency is fixed or floating
QUIET = False
DEBUG = False

#1. Create an instance of Acdc
#2. calibrate_board()
#3. process_files(file_list), where file_list is raw data files
#4.	test_acdc.save_npz('acdc52_wrtimes_only_094355')

	# test_acdc.load_npz('acdc62_stat2_full')
# Globals for debugging purposes only
all_xh = []
all_yh = []

if DEBUG:
	MAX_PROCESSES = 1

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

def compute_sliding_function(xdata, ydata, lbound, rbound, stat_spline, func, slide_increment=1, FAST=False):
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

	if FAST:
		abs_ydata = np.absolute(ydata)
		ydata_max = np.amax(abs_ydata)
		indices, _ = find_peaks(abs_ydata, height=0.6*ydata_max, distance=5)
		limit = indices[-1] - indices[0]
		if limit == 0:
			limit = 256
		else:
			limit += 15
	else:
		limit = 256

	func_vals = []
	lags = []
	lag = 0
	lag_factor = 25/256
	while lag < limit:

		xdata_sliding = np.copy(xdata) - lag_factor*lag*np.ones_like(xdata)
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

		# if round(lag_factor*lag, 3) == 1.66:
		# 	fig, ax = plt.subplots()
		# 	ax.scatter(xdata, ydata, marker='.', label='Stationary')
		# 	ax.plot(domain_for_plot, stat_spline(domain_for_plot), label='Stationary spline')
		# 	ax.scatter(xdata_shifted_inbounds, ydata_inbounds, marker='.', label='Lagged')
		# 	ax.scatter(xdata_shifted_inbounds, ydata_stationary, marker='.', color='red')
		# 	for i, x_val in enumerate(xdata_shifted_inbounds):
		# 		if ydata_inbounds[i] > ydata_stationary[i]:
		# 			lower_val = ydata_stationary[i]
		# 			higher_val = ydata_inbounds[i]
		# 		else:
		# 			lower_val = ydata_inbounds[i]
		# 			higher_val = ydata_stationary[i]
		# 		ax.axvline(x_val, ymin=(lower_val+1200)/1100, ymax=(higher_val+1200)/1100, color='gray')
		# 	ax.axvline(lbound, color='green')
		# 	ax.axvline(rbound, color='green')
		# 	# ax.axvline(xdata_shifted_inbounds[3], color='C1')
		# 	ax.text(2.663, -1180, f'Lag: {round(lag_factor*lag, 3)} ns', fontdict=dict(size=12))
		# 	ax.set_xlim(lbound-0.2, rbound+0.2)
		# 	# ax.set_xlim(1.5, 9)
		# 	ax.set_ylim(-1200, -100)
		# 	ax.legend(loc='lower right', framealpha=1)
		# 	ax.set_xlabel('Sample time (ns)')
		# 	ax.set_ylabel('ADC count')
		# 	ax.xaxis.set_ticks_position('both')
		# 	ax.yaxis.set_ticks_position('both')
		# 	plt.minorticks_on()

		# 	fig2, ax2 = plt.subplots()
		# 	ax2.scatter(xdata_shifted_inbounds, (ydata_inbounds-ydata_stationary)**2, color='black', marker='.')
		# 	for i in range(len(xdata_shifted_inbounds)-1):
		# 		x_pols = [xdata_shifted_inbounds[i], xdata_shifted_inbounds[i], xdata_shifted_inbounds[i+1], xdata_shifted_inbounds[i+1]]
		# 		y_pols = [0, ((ydata_inbounds-ydata_stationary)**2)[i], ((ydata_inbounds-ydata_stationary)**2)[i+1], 0]
		# 		ax2.fill(x_pols, y_pols, color='C0', alpha=0.45, edgecolor='black')
		# 	ax2.text(2.813,5.35e5, f'Lag: {round(lag_factor*lag, 3)} ns', fontdict=dict(size=12))
		# 	ax2.text(2.813, 4.9e5, 'Integral value: {:.2e}'.format(func_vals[-1]), fontdict=dict(size=12))
		# 	ax2.set_xlabel('Sample time (ns)')
		# 	ax2.set_ylabel('Least squares values')
		# 	ax2.xaxis.set_ticks_position('both')
		# 	ax2.yaxis.set_ticks_position('both')
		# 	plt.minorticks_on()

		# 	plt.show()

		lag += slide_increment

	func_vals = np.array(func_vals)
	lags = np.array(lags)

	return lags, func_vals

def find_leading_edge(xdata, ydata, display, SPLINE_CFD=False):

	# Determines the indices of the peaks in the prompt and reflected pulses
	height_cutoff = -0.6*ydata.max()
	distance_between_peaks = 20		# in units of indices
	peak_region_radius = 15			# in units of indices
	# removed -1*ydata for vcc calibrated stuff
	peaks_rough = find_peaks(ydata, height=height_cutoff, distance=distance_between_peaks)[0]
	prompt_peak_index, reflect_peak_index = np.sort(peaks_rough[ydata[peaks_rough].argsort()[0:2]])

	# Creates subregions of data around the reflect peak
	reflect_lbound = reflect_peak_index - int((reflect_peak_index-prompt_peak_index)/2)-5 # lower bound is a bit left of the midway between peaks
	reflect_ubound = reflect_peak_index + 6
	ydata_subrange = ydata[reflect_lbound:reflect_ubound]
	reflect_subdomain = xdata[reflect_lbound:reflect_ubound]
	peak_region_lower, peak_region_upper = xdata[reflect_peak_index-4], xdata[reflect_peak_index+4]

	# Solves for the extrema of the reflect peak
	# spline_tuple = splrep(reflect_subdomain, ydata_subrange, k=3, s=10000)
	spline_tuple = splrep(reflect_subdomain, ydata_subrange, k=3)
	reflect_bspline = BSpline(*spline_tuple)
	reflect_dbspline = reflect_bspline.derivative()
	reflect_dcubic_spline = CubicSpline(reflect_subdomain, reflect_dbspline(reflect_subdomain))
	extrema = reflect_dcubic_spline.solve(0, extrapolate=False)
	fig, ax = plt.subplots()
	ax.plot(reflect_subdomain, reflect_bspline(reflect_subdomain))
	for thing in extrema:
		ax.axvline(thing)
	ax.axvline(peak_region_lower, label='lower', color='red')
	ax.axvline(peak_region_upper, label='upper', color='purple')
	ax.legend()
	plt.show()
	print(extrema)
	reflect_peak_max = reflect_bspline(extrema[(extrema > peak_region_lower) & (extrema < peak_region_upper)])	# finds the extrema that is near our original find_peaks value
	reflect_peak_max = reflect_peak_max[0]
	reflect_peak_min_val = reflect_bspline(extrema[0]) + 0.1*(reflect_peak_max - reflect_bspline(extrema[0]))

	# repeating the spline for the prompt peak now
	prompt_lbound = prompt_peak_index - 20
	if prompt_lbound < 0:
		prompt_lbound = 0
	prompt_ubound = prompt_peak_index + 4
	prompt_subrange = ydata[prompt_lbound:prompt_ubound]
	prompt_subdomain = xdata[prompt_lbound:prompt_ubound]
	# prompt_tuple = splrep(prompt_subdomain, prompt_subrange, k=3, s=10000)
	prompt_tuple = splrep(prompt_subdomain, prompt_subrange, k=3)
	prompt_bspline = BSpline(*prompt_tuple)
	prompt_cubic_spline = CubicSpline(prompt_subdomain, prompt_bspline(prompt_subdomain))
	prompt_dbspline = prompt_bspline.derivative()
	prompt_dcubic_spline = CubicSpline(prompt_subdomain, prompt_dbspline(prompt_subdomain))
	prompt_extrema = prompt_dcubic_spline.solve(0)
	peak_region_lower, peak_region_upper = xdata[prompt_peak_index-3], xdata[prompt_peak_index+3]
	prompt_peak_max = prompt_bspline(prompt_extrema[(prompt_extrema > peak_region_lower) & (prompt_extrema < peak_region_upper)])
	prompt_peak_max = prompt_peak_max[0]

	fig, ax = plt.subplots()
	ax.scatter(xdata, ydata)
	ax.plot(prompt_subdomain, prompt_bspline(prompt_subdomain))
	ax.axhline(reflect_peak_min_val, color='red', label='min')
	ax.axhline(0.9*prompt_peak_max, color='purple', label='max')
	ax.legend()
	plt.show()

	# Computes the integral bounds
	lbound = prompt_cubic_spline.solve(reflect_peak_min_val, extrapolate=False)[0]
	rbound = prompt_cubic_spline.solve(0.9*prompt_peak_max, extrapolate=False)[0]

	if display:
		fig3, ax3 = plt.subplots()
		ax3.scatter(xdata, ydata, marker='.', label='Raw data')

		# reflect_peak_spline_domain = np.linspace(xdata[reflect_lbound], xdata[reflect_ubound-1], 100)
		# ax3.plot(reflect_peak_spline_domain, reflect_bspline(reflect_peak_spline_domain), color='orange', label='Reflected Pulse Spline')

		prompt_peak_spline_domain = np.linspace(xdata[prompt_lbound], xdata[prompt_ubound-1], 100)
		ax3.plot(prompt_peak_spline_domain, prompt_bspline(prompt_peak_spline_domain), color='green')
		# ax3.plot(2.22, reflect_peak_min_val, marker='o', color='red', label='intersection')

		# ax3.axhline(reflect_peak_min_val, color='magenta', label=f'{round(100*reflect_peak_min_val/reflect_peak_max, 2)}% of reflect peak max')
		ax3.axvline(lbound, color='red', label=f'Lower bound ({round(100*reflect_peak_min_val/prompt_peak_max, 2)}% of \nprompt peak max)')
		ax3.axvline(rbound, color='purple', label='Upper bound (90% of \nprompt peak max)')
		ax3.legend(loc='lower right')
		ax3.set_xlabel('Sample time (ns)')
		ax3.set_ylabel('ADC Count')
		ax3.set_title('Reflection-dependent Leading Edge Bounds')
		ax3.set_xlim(1.6, 9.1)
		plt.show()
	
	if SPLINE_CFD:
		reflect_cubicspline = CubicSpline(reflect_subdomain, reflect_bspline(reflect_subdomain))
		r_intersects = reflect_cubicspline.solve(reflect_peak_min_val, extrapolate=False)
		reflect_cfd_pos = (r_intersects[r_intersects < xdata[reflect_peak_index]])[-1]
		return lbound, reflect_cfd_pos
	else:
		return lbound, rbound, prompt_cubic_spline

@numba.jit(nopython=True)
def gauss_const_back(x, A, c, mu, B):
	return A*np.exp(-c*(x-mu)**2) + B

@numba.jit(nopython=True)
def sin_const_back(x, A, omega, phi, B):
	return A*np.sin(omega*x-phi)+B

@numba.jit(nopython=True)
def sin_const_back_250(x, A, phi, B):
	return A*np.sin(2*np.pi*0.25*x-phi)+B

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
	def __init__(self, config_data):
		"""Initializes based on a python dict or a yaml file
		"""

		# Loads configuration file. First checks if it's a yaml file; otherwise treats as Python dict
		if isinstance(config_data, str):
			try:
				with open(config_data, 'r') as yf:
					config_data = yaml.safe_load(yf)
			except FileNotFoundError:
				print(f'{config_data} doesn\'t exist')
				config_data = None
		elif not isinstance(config_data, dict):
			print(f'`config_data` file-type not recognized: {type(config_data)}')
			config_data = None

		#not used yet, but I'm creating a self attribute to start the migration
		#of the config_data into something that is more... compact
		self.c = config_data

		# Constants
		self.chan_rearrange = np.array([5,4,3,2,1,0,11,10,9,8,7,6,17,16,15,14,13,12,23,22,21,20,19,18,29,28,27,26,25,24])
		self.vel = 144.					# mm/ns
		self.dt = 1.0/(40e6*256)*1e9	# PLL is 40 MHz, DLL is 256 samples, nominal sample spacing = 1/(256*40 MHz)
				
		# Metadata
		self.acdc_id = config_data['acdc_id']			# ACDC number (see inventory), e.g. '46'
		self.lappd_id = config_data['lappd_id']			# Incom manufacturing number, e.g. '125'
		self.acc_id = config_data['acc_id']				# ACC nmber (see inventory), e.g. '1'
		self.station_id = config_data['station_id']		# station position, e.g. '1'
		self.sync_ch = self.chan_rearrange[config_data['sync_ch']]

		# Positioning data
		self.zpos = config_data['zpos']						# Distance to next upstream station
		self.corner_offset = config_data['corner_offset']	# Distance corner of active area is from beam axis

		# Pedestal related
		self.ped_data_path = config_data['pedestal_file_name']
		self.pedestal_data = None
		self.pedestal_counts = None

		# Calibration related
		self.calib_data_file_path = config_data['calib_file_name']
		self.vccs = None
		self.reflect_time_offset = np.full(30, 3.3)
		self.wr_calib_offset = np.full(30, 3.26)
		self.strip_pos = 6.6*np.linspace(0,29,30)	

		# Derived data
		self.waveforms_optch = None
		self.waveforms_sin = None
		self.hpos = None
		self.vpos = None	

		# self.pedestal_counts = init_data_dict['pedestal_counts']	# ADC count; a list of 256 integers, which corresponds to each capicitors of VCDL.
		# self.pedestal_voltage = init_data_dict['pedestal_voltage']
		
		if not QUIET:
			print(f'ACDC instantiated\n  ACDC:    {self.acdc_id}\n  LAPPD:   {self.lappd_id}\n  ACC:     {self.acc_id}\n  Station: {self.station_id}\n')

	def import_raw_data(self, raw_data_path, is_pedestal_data=False):
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

		# Status update about which data file we're importing
		if not QUIET:
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
		times = np.array(times)
		times = ((times>>32) & 0xffffffff) + 1e-9*(4*(times & 0xffffffff))
		
		data = np.array(data).reshape([-1,30,256])

		data = data[:,self.chan_rearrange,:]

		if is_pedestal_data:
			# Averages pedestal_data over all the events
			self.pedestal_data = np.copy(data)
			self.pedestal_counts = np.copy(self.pedestal_data.mean(0))

		# Still returns relevant data
		return times_320, times, data

	def calibrate_board(self):

		vccs = [[None]*256 for i in range(30)]
		if CALIB_ADC:
			# Imports voltage calib data and normalizes
			with uproot.open(self.calib_data_file_path) as calib_file:

				# Gets numpy array axes are: channel, cap, voltage increment, and ADC type
				voltage_counts = np.reshape(calib_file["config_tree"]["voltage_count_curves"].array(library="np"), (30,256,256,2))
				voltage_counts[:,:,:,0] = voltage_counts[:,:,:,0]*1.2/4096.

				# Filter the data and make it monotonically increasing
				# voltage_counts[:,:,:,1] = savgol_filter(voltage_counts[:,:,:,1], 41, 2, axis=2)
				# reorder = np.argsort(voltage_counts[:,:,:,0], axis=2)
				# voltage_counts = np.take_along_axis(voltage_counts, reorder[:,:,:,np.newaxis], axis=2)
				# warnings.simplefilter('error')
				for ch in range(0, 30):
					for cap in range(0, 256):
						# print(f'{ch} {cap}')

						vert_mask = np.append((np.diff(voltage_counts[ch,cap,:,1]) > 4), False)
						vert_mask = vert_mask & np.roll(vert_mask, 1)
						adc_data = voltage_counts[ch, cap, vert_mask, 1]
						volt_data = voltage_counts[ch, cap, vert_mask, 0]

						tck = splrep(adc_data, volt_data, s=0.00005, k=3)
						single_bspline = BSpline(*tck, extrapolate=True)
						vccs[ch][cap] = single_bspline

						# if (ch == 3 and cap == 5) or (ch == 23 and cap == 69):
						# fig, ax = plt.subplots()
						# ax.scatter(voltage_counts[ch,cap,:,1], voltage_counts[ch,cap,:,0], marker='.', color='black')
						# fig_domain = np.linspace(0, 4096, 1000)
						# ax.plot(fig_domain, vccs[ch][cap](fig_domain), color='red')
						# plt.show()

				vccs = [vccs[i] for i in self.chan_rearrange]				

			if not QUIET:
				print('ACDC voltage calibrated with VCCs')
		
		else:
			self.import_raw_data(self.ped_data_path, is_pedestal_data=True)

			if not QUIET:
				print('ACDC ADC counts calibrated with pedestal data')

		self.vccs = vccs

		if CALIB_TIME_BASE:
			# Creates a time-base calibrated array of sample times
			with uproot.open(self.calib_data_file_path) as calib_file:
				time_base = np.reshape(calib_file['config_tree']['time_offsets'].array(library='np'), (30,256))
				time_base = time_base[self.chan_rearrange,:]*1e9

			if not QUIET:
				print('ACDC sample times calibrated\n')

		else:
			time_base = np.full(256, self.dt) #+ np.random.normal(loc=0., scale=0.010, size=256)
			time_base[-1] += 0.300
			time_base = np.tile(time_base, 30).reshape(30, 256)
			# time_base[:,-1] -= (np.sum(time_base, axis=1) - 25)	# normalizes cumulative samples to 25 ns
			# if (time_base[:,-1] < 0).any():
			# 	print('Error: wrap-around offset less than 0')
			# 	exit()

			if not QUIET:
				print('ACDC sample times NOT calibrated\n')

		self.time_base = time_base

		return

	def preprocess_data(self, data_raw, times_320):

		# Calibrates the ADC data
		data = np.empty_like(data_raw, dtype=np.float64)
		if CALIB_ADC:
			for ch in range(0,30):
				for cap in range(0,256):
					data[:,ch,cap] = self.vccs[ch][cap](data_raw[:,ch,cap])
		else:
			data = data_raw - self.pedestal_counts

		# Selects optimal y data (voltage) and channels
		if NO_POSITIONS:
			ydata_v, opt_chs, misfire_masks = np.full((times_320.shape[0], 256), 0), np.full(times_320.shape[0], 0), np.full((times_320.shape[0], 256), True)
		else:
			ydata_v, opt_chs, misfire_masks = self.v_data_opt_ch(data) # Here we select one channel to be the optimal channel of this event to be analyzed
		xdata_v = np.tile(np.copy(self.strip_pos), ydata_v.shape[0]).reshape(ydata_v.shape[0], 30)
		# xdata_v = np.tile(np.delete(np.copy(self.strip_pos), self.sync_ch), ydata_v.shape[0]).reshape(ydata_v.shape[0], 29)

		num_events = opt_chs.shape[0]

		# Adjusts trigger position
		trigger_low = (((times_320+2+2)%8)*32-16)%256
		# trigger_high = (((times_320+2+2)%8)*32+24)%256
	
		xdata_h = np.copy(self.time_base)[opt_chs,:]
		xdata_h = np.array([np.roll(np.copy(xdata_h[i,:]), -trigger_low[i]) for i in range(num_events)])
		xdata_h = np.roll(xdata_h, 1, axis=1)
		xdata_h = np.cumsum(xdata_h, axis=1)
		ydata_h = np.take_along_axis(data, opt_chs[:,np.newaxis,np.newaxis], axis=1).reshape(num_events, 256)
		ydata_h = np.array([np.roll(np.copy(ydata_h[i,:]), -trigger_low[i]) for i in range(num_events)])
		misfire_masks = np.array([np.roll(np.copy(misfire_masks[i,:]), -trigger_low[i]) for i in range(num_events)])

		optchs_sine = np.full(num_events, self.sync_ch)
		xdata_sine = np.copy(self.time_base)[optchs_sine,:]
		xdata_sine = np.array([np.roll(np.copy(xdata_sine[i,:]), -trigger_low[i]) for i in range(num_events)])
		xdata_sine = np.roll(xdata_sine, 1, axis=1)
		xdata_sine = np.cumsum(xdata_sine, axis=1)
		ydata_sine = np.take_along_axis(data, optchs_sine[:,np.newaxis,np.newaxis], axis=1).reshape(num_events, 256)
		ydata_sine = np.array([np.roll(np.copy(ydata_sine[i,:]), -trigger_low[i]) for i in range(num_events)])

		# xdata_h = np.tile(np.linspace(0,255,256,dtype=int), num_events).reshape(num_events, 256)
		# xdata_sine = np.copy(xdata_h)

		if DEBUG:
			global all_xh
			global all_yh
			all_xh = np.repeat(np.copy(self.time_base)[np.newaxis,:,:], num_events, axis=0)
			all_xh = np.array([np.roll(np.copy(all_xh[i,:,:]), -trigger_low[i], axis=0) for i in range(num_events)])
			all_xh = np.roll(all_xh, 1, axis=2)
			all_xh = np.cumsum(all_xh, axis=2)
			all_yh = np.array([np.roll(np.copy(data[i,:,:]), -trigger_low[i], axis=1) for i in range(num_events)])
			# all_yh = np.copy(data)
			# all_xh = np.tile(np.tile(np.linspace(0,255,256,dtype=int),30), num_events).reshape(num_events, 30, 256)
   
		# for event in [500,1500,2500,3500,4500,5500,6500,7500,8500,9500]:
		# 	fig, ax = plt.subplots()
		# 	ax.set_title(f'Event {event}')
		# 	wrap = xdata_sine[event, :][np.append(np.diff(xdata_sine[event, :]) > 0.300, False)]
		# 	ax.plot(xdata_sine[event, :], ydata_sine[event, :])
		# 	ax.axvline(wrap)
		# 	plt.show()

		return xdata_v, ydata_v, xdata_h, ydata_h, opt_chs, misfire_masks, xdata_sine, ydata_sine, trigger_low

	def v_data_opt_ch(self, data):
		"""Small function to retrieve the channel to be used in to find the x-positions.
		Arguments:
			(Acdc) self
			(np.array) waveform: a single event's waveform (2D array) from which the optimal channel will be found		
		"""

		dummy_data = np.copy(data)
		# dummy_data = np.delete(dummy_data, self.sync_ch, axis=1)

		IMPROVED = True
		if IMPROVED:
			# Finds optimal channels (1D array, length=# of events, best ch per event)
			if CALIB_ADC:
				baseline = 0.85
				too_low_val = 0.15
			else:
				baseline = 0
				too_low_val = -2700
			dummy_data[dummy_data < too_low_val] = baseline
			ch_mins = dummy_data.min(axis=2)
			organized_chs = np.argsort(ch_mins, axis=1)
			opt_chs = organized_chs[:,0]
			misfire_masks = np.take_along_axis(np.copy(data), opt_chs[:,np.newaxis,np.newaxis], axis=1).reshape((data.shape[0], 256)) >= too_low_val
		else:
			# Finds optimal channels (1D array, length=# of events, best ch per event)
			ch_mins = dummy_data.min(axis=2)
			organized_chs = np.argsort(ch_mins, axis=1)
			opt_chs = organized_chs[:,0]
			misfire_masks = np.full((10000,256), True)
		
		return ch_mins, opt_chs, misfire_masks

	def calc_positions(self, xdata_v, ydata_v, xdata_h, ydata_h, opt_chs, misfire_masks, xdata_sine, ydata_sine, trigger_low, times):

		max_offset = 10
		offset_increment = 0.1
		offsets = np.arange(0, max_offset, offset_increment)

		A_vec = []
		chi2_vec = []
		delta_t_vec = []
		vpos_vec = []
		phi_vec = []
		first_peak_vec = []
		omega_vec = []
		startcap_vec = []

		# Restricting sin data bounds to exclude trigger samples
		sin_lbound, sin_rbound = int(4*(256/25)), int(21*(256/25))
		cut = np.linspace(sin_lbound, sin_rbound, sin_rbound-sin_lbound+1, dtype=int)
		xdata_sine_cut, ydata_sine_cut = xdata_sine[:,cut], ydata_sine[:,cut]

		# Vectorized p0 for sin fit
		B0 = np.average(ydata_sine, axis=1)
		A0 = np.max(ydata_sine, axis=1) - B0
		omega0 = np.full_like(B0, 2*np.pi*0.25)
		phi0 = np.zeros_like(B0)
		p0_array = np.array([A0, omega0, phi0, B0]).T
		
		skipped = []
		for i, (xv, yv, xh, yh, opt_ch, misfire_mask, xsin, ysin, p0, tl) in enumerate(zip(xdata_v, ydata_v, xdata_h, ydata_h, opt_chs, misfire_masks, xdata_sine_cut, ydata_sine_cut, p0_array, trigger_low)):
			try:

				wraparound_ind = 255-tl
				startcap = tl

				xh, yh = xh[misfire_mask], yh[misfire_mask]	
				
				# Finds spatial position
				if NO_POSITIONS:
					delta_t, lbound, vpos = 0, 0, 0
				else:
					delta_t, lbound, reflect_ind = self.calc_delta_t(xh, yh, offsets)
					if wraparound_ind < reflect_ind:	# if wraparound is between peaks throw out event
						raise
					mu0 = xv[opt_ch]
					vpos = self.calc_vpos(xv, yv, mu0)
				
				# Excludes wraparound in the sine fit
				if EXCLUDE_WRAP:
					# Keeps samples to left of wraparound
					if wraparound_ind >= 128 and wraparound_ind < sin_rbound:		
						xsin, ysin = xsin[:wraparound_ind-sin_lbound], ysin[:wraparound_ind-sin_lbound]
					# Keeps samples to right of wraparound
					elif wraparound_ind < 128 and wraparound_ind > sin_lbound:		
						xsin, ysin = xsin[wraparound_ind-sin_lbound+1:], ysin[wraparound_ind-sin_lbound+1:]
					# Must fit a full period
					if len(xsin) < 40:
						raise

				# Gets rid of any misfired caps with super low ADC value
				badcap_cut = ysin > 0.2
				xsin, ysin = xsin[badcap_cut], ysin[badcap_cut]
				
				# Fits sine with variable or constant (250 MHz) frequency
				if NO_SINES:
					popt = [0., 0., 0., 0.]
					pcov = [0.]
				elif VAR_SINE_FREQ:
					param_bounds = ([0.025, 1.4, -3*np.pi, 0.6], [0.4, 1.75, 3*np.pi, 0.9])
					popt, pcov = curve_fit(sin_const_back, xsin, ysin, p0=p0, bounds=param_bounds)
				else:
					popt, pcov = curve_fit(sin_const_back_250, xsin, ysin, p0=(p0[0], p0[2], p0[3]), bounds=([0.025, -3*np.pi, 0.6], [0.4, 3*np.pi, 0.9]))
					popt = [popt[0], 2*np.pi*0.25, popt[1], popt[2]]
				
				# Temporary fit variables for debugging, since the fit gives omega in rad/ns and phi in rad
				# tempomega = popt[1]/(2*np.pi)*1e3
				# tempphi = (popt[2]%(2*np.pi))/(2*np.pi*0.25)
				# if tempphi>=2: tempphi -= 4.

				# Currently not good method of getting goodness-of-fit
				sinsigma = np.sqrt(np.diag(pcov))
				# r = ysin - sin_const_back(xsin, *popt)
				chi2 = sinsigma
				# chi2 = r.T @ np.linalg.inv(sinsigma) @ r

				# fig, ax = plt.subplots()
				# ax.scatter(xh, yh, marker='.', color='black', label='Raw Data')
				# ax.plot(xh, yh, color='black')
				# ax.set_xlabel('Time (ns)', fontdict=dict(size=14))
				# ax.set_ylabel('Voltage (V)', fontdict=dict(size=14))
				# ax.xaxis.set_ticks_position('both')
				# ax.yaxis.set_ticks_position('both')
				# plt.minorticks_on()
				# plt.show()
				
				A_vec.append(popt[0])
				chi2_vec.append(chi2)
				delta_t_vec.append(delta_t)
				vpos_vec.append(vpos)	
				phi_vec.append(popt[2])
				first_peak_vec.append(lbound)
				omega_vec.append(popt[1])
				startcap_vec.append(startcap)

			except Exception as err:
				skipped.append(i)	
				if DEBUG:
					raise err
		
		num_skipped = len(skipped)

		times_wr_vec = np.delete(times, skipped)

		chi2_vec = np.array(chi2_vec)

		A_vec = np.array(A_vec)
		phi_vec = np.array(phi_vec)
		phi_vec[A_vec < 0] -= np.pi
		phi_vec = phi_vec%(2*np.pi)
		phi_vec = phi_vec/(2*np.pi*0.25)
		phi_vec[phi_vec >= 2.] -= 4.

		omega_vec = np.array(omega_vec)

		delta_t_vec = np.array(delta_t_vec)
		first_peak_vec = np.array(first_peak_vec)
		hpos_vec = 0.5*self.vel*(delta_t_vec - np.delete(self.reflect_time_offset[opt_chs], skipped))

		calib_constants = np.delete(self.wr_calib_offset[opt_chs], skipped)
		eventphi_vec = (first_peak_vec - (200./self.vel - hpos_vec/self.vel)) - phi_vec	# 200 mm = length of LAPPD

		opt_chs = np.delete(opt_chs, skipped)
		
		startcap_vec = np.array(startcap_vec)
		
		return hpos_vec, vpos_vec, times_wr_vec, eventphi_vec, first_peak_vec, phi_vec, omega_vec, delta_t_vec, opt_chs, chi2_vec, startcap_vec, num_skipped

	def calc_vpos(self, xv, yv, mu0):

		p0 = [-0.25*yv.max(), 0.01, mu0, 0.8]
		popt, pcov = curve_fit(gauss_const_back, xv, yv, p0=p0)
		# fig, ax = plt.subplots()
		# ax.scatter(xv, yv, marker='.', color='black')
		# domain = np.linspace(xv[0], xv[-1], 200)
		# ax.plot(domain, gauss_const_back(domain, *p0), color='green')
		# ax.plot(domain, gauss_const_back(domain, *popt), color='red')
		# plt.show()

		return popt[2]
	
	def calc_delta_t(self, xh, yh, offsets, debug=False):
		"""
		Returns the time difference between the two peaks in the waveform.
		"""

		lbound, rbound, reflect_ind = self.leading_edge_bounds(xh, yh)

		lsquares = self.find_lsquares(xh, yh, lbound, rbound, offsets)

		cut = (offsets > 3) & (offsets < 9)
		peak_rough = offsets[cut][lsquares[cut].argmin()]

		fitcut = (offsets > (peak_rough - 0.5)) & (offsets < (peak_rough + 0.5))
		offsets_cut = offsets[fitcut]
		lsquares_cut = lsquares[fitcut]
		spline_tup = splrep(offsets_cut, lsquares_cut, k=4)
		bspline = BSpline(*spline_tup)
		dbspline = bspline.derivative()

		ppoly = PPoly.from_spline(dbspline)
		extrema = ppoly.roots(extrapolate=False)

		extrema = extrema[(extrema > peak_rough - 0.2) & (extrema < peak_rough + 0.2)]

		if len(extrema) > 0:
			delta_t = extrema[bspline(extrema).argsort()][-1]
		else:
			delta_t = peak_rough

		if debug:
			fig, ax = plt.subplots()
			ax.scatter(offsets, lsquares, marker='.', color='black')
			domain = np.linspace(offsets_cut[0], offsets_cut[-1], 250)
			ax.plot(domain, bspline(domain), color='red')
			ax.axvline(delta_t, color='green')

			fig, ax = plt.subplots()
			ax.scatter(xh, yh, marker='.', color='black')
			ax.axvline(lbound)
			ax.axvline(rbound)
			plt.show()

		return delta_t, lbound, reflect_ind

	def leading_edge_bounds(self, xh, yh):

		yh_temp = -yh + yh.max()
		min_height = 0.6*yh_temp.max()
		peak_dist = 20
		
		peaks_rough = find_peaks(yh_temp, height=min_height, distance=peak_dist)[0]
		prompt_ind, reflect_ind = peaks_rough[peaks_rough > 8][0:2]

		lbound = prompt_ind - 25
		if lbound < 0:
			lbound = 0
		rbound = prompt_ind + 4
		subdomain = xh[lbound:rbound]
		subrange = yh[lbound:rbound]
		cspline = CubicSpline(subdomain, subrange, extrapolate=False, bc_type='natural')

		ymin, ymax = yh[lbound], yh[prompt_ind]
		lbound_y = ymin - 0.1*(ymin-ymax)
		rbound_y = ymin - 0.9*(ymin-ymax)
		
		lbound = cspline.solve(lbound_y, extrapolate=False)[0]
		rbound = cspline.solve(rbound_y, extrapolate=False)[0]

		# fig, ax = plt.subplots()
		# ax.scatter(xh, yh, marker='.', color='black')
		# domain = np.linspace(subdomain[0], subdomain[-1], 100)
		# ax.plot(domain, cspline(domain), color='red')
		# ax.axvline(lbound, color='blue')
		# ax.axvline(rbound, color='blue')
		# plt.show()

		# tck = splrep(subdomain, subrange, s=0.0005)
		# bspline = BSpline(*tck)

		# fig, ax = plt.subplots()
		# ax.scatter(xh, yh, marker='.', color='black')
		# ax.plot(domain, bspline(domain), color='green')
		# ax.axvline(lbound, color='blue')
		# ax.axvline(rbound, color='blue')
		# plt.show()

		return lbound, rbound, reflect_ind

	def find_lsquares(self, xh, yh, lbound, rbound, offsets):

		bspline_tup = splrep(xh, yh, k=3)
		bspline = BSpline(*bspline_tup)

		x = np.linspace(lbound, rbound, 10)
		y = bspline(x)

		x_shift = np.vstack([x + dt for dt in offsets])
		y_shift = bspline(x_shift)

		least_squares = (y_shift - y)**2
		avg_lsquares = trapezoid(least_squares, x, axis=1)

		return avg_lsquares

	def process_single_file(self, file_name):
		times_320, times, data_raw = self.import_raw_data(file_name)
		preprocess_vec = self.preprocess_data(data_raw, times_320)
		waveforms_optch = np.array([preprocess_vec[2], preprocess_vec[3]])
		waveforms_sin = np.array([preprocess_vec[6], preprocess_vec[7]])
		analyzed_vec = self.calc_positions(*preprocess_vec, times)
		return analyzed_vec, waveforms_optch, waveforms_sin

	#TODO: add an argument to this which is a list of pedestal files. 
	#Then, with each data file, look at when the last pedestal calibration was done. 
	#If there is a closer-in-time pedestal file, recalibrate with that. 
	def process_files(self, file_list):

		waveforms_optch_vec, waveforms_sin_vec = np.empty((2, 0, 256), dtype=np.float64), np.empty((2, 0, 256), dtype=np.float64)
		hpos_vec, vpos_vec, times_wr_vec, eventphi_vec, first_peak_vec, phi_vec, omega_vec, delta_t_vec, opt_chs_vec, chi2_vec, startcap_vec, total_skipped = [], [], [], [], [], [], [], [], [], [], [], 0
		t1 = process_time()
		t3 = time()
		if MAX_PROCESSES != 1:
			with Pool(MAX_PROCESSES) as p:
				file_list = convert_to_list(file_list)
				rv = p.map(self.process_single_file, file_list)

			for analyzed_vec, waveforms_optch, waveforms_sin in rv:
				hpos, vpos, times_wr, eventphi, first_peak, phi, omega, delta_t, opt_chs, chi2, startcap, num_skipped = analyzed_vec
				waveforms_optch_vec = np.append(waveforms_optch_vec, waveforms_optch, axis=1)
				waveforms_sin_vec = np.append(waveforms_sin_vec, waveforms_sin, axis=1)
				hpos_vec.extend(hpos)
				vpos_vec.extend(vpos)
				times_wr_vec.extend(times_wr)
				eventphi_vec.extend(eventphi)
				first_peak_vec.extend(first_peak)
				phi_vec.extend(phi)
				omega_vec.extend(omega)
				delta_t_vec.extend(delta_t)
				opt_chs_vec.extend(opt_chs)
				chi2_vec.extend(chi2)
				startcap_vec.extend(startcap)
				total_skipped += num_skipped

		else:
			for file_name in file_list:
				analyzed_vec, waveforms_optch, waveforms_sin = self.process_single_file(file_name)
				hpos, vpos, times_wr, eventphi, first_peak, phi, omega, delta_t, opt_chs, chi2, startcap, num_skipped = analyzed_vec
				waveforms_optch_vec = np.append(waveforms_optch_vec, waveforms_optch, axis=1)
				waveforms_sin_vec = np.append(waveforms_sin_vec, waveforms_sin, axis=1)
				hpos_vec.extend(hpos)
				vpos_vec.extend(vpos)
				times_wr_vec.extend(times_wr)
				eventphi_vec.extend(eventphi)
				first_peak_vec.extend(first_peak)
				phi_vec.extend(phi)
				omega_vec.extend(omega)
				delta_t_vec.extend(delta_t)
				opt_chs_vec.extend(opt_chs)
				chi2_vec.extend(chi2)
				startcap_vec.extend(startcap)
				total_skipped += num_skipped

		t2 = process_time()
		t4 = time()

		print(f'Total skipped: {round(100.*total_skipped/(total_skipped + len(hpos_vec)),2)}%')
		print(f'Total events analyzed: {total_skipped + len(hpos_vec)}')
		print(f'Process time duration: {round(t2-t1, 3)} s')
		print(f'Wall clock duration: {round(t4-t3, 3)} s')

		self.waveforms_optch = waveforms_optch
		self.waveforms_sin = waveforms_sin
		self.hpos = np.array(hpos_vec)
		self.vpos = np.array(vpos_vec)
		self.times_wr = np.array(times_wr_vec)
		self.eventphi= np.array(eventphi_vec)
		self.first_peak = np.array(first_peak_vec)
		self.phi = np.array(phi_vec)
		self.omega = np.array(omega_vec)
		self.delta_t = np.array(delta_t_vec)
		self.opt_chs = np.array(opt_chs_vec)
		self.chi2 = np.array(chi2_vec)
		self.startcap = np.array(startcap_vec)
		self.skipped = total_skipped
		return
	
	def plot_events(self, file_list):

		for file_name in file_list:
			times_320, times, data_raw = self.import_raw_data(file_name)
			preprocess_vec = self.preprocess_data(data_raw, times_320)
			for i in range(len(times)):
				fig, ax = plt.subplots()
				for chan in range(0,30):
					ax.plot(all_xh[i,chan], all_yh[i,chan])
				plt.show()

		return

	def plot_vccs(self, ch=-1, cap=-1):

		if ch < 0:
			ch = int(30*np.random.rand())
		if cap < 0:
			cap = int(256*np.random.rand())

		with uproot.open(self.calib_data_file_path) as calib_file:

			# Gets numpy array axes are: channel, cap, voltage increment, and ADC type
			voltage_counts = np.reshape(calib_file["config_tree"]["voltage_count_curves"].array(library="np"), (30,256,256,2))
			voltage_counts[:,:,:,0] = voltage_counts[:,:,:,0]*1.2/4096.

			# Filter the data and make it monotonically increasing
			# voltage_counts[:,:,:,1] = savgol_filter(voltage_counts[:,:,:,1], 41, 2, axis=2)	
			# reorder = np.argsort(voltage_counts[:,:,:,0], axis=2)
			# voltage_counts = np.take_along_axis(voltage_counts, reorder[:,:,:,np.newaxis], axis=2)

			voltage_counts = voltage_counts[self.chan_rearrange,:,:,:]
			voltage_counts = voltage_counts[ch,cap,:,:]
			fig, ax = plt.subplots()
			ax.scatter(voltage_counts[:,1],voltage_counts[:,0], marker='.', color='black', label=f'Ch: {ch}, cap: {cap}')
			domain = np.linspace(0, 4096, 500)
			ax.plot(domain, self.vccs[ch][cap](domain), color='red', label='Fit')
			# ax.axhline(voltage_counts[60, 0], color='green', label='Fit bounds')
			# ax.axhline(voltage_counts[196, 0], color='green')
			ax.legend()
			ax.set_xlabel('ADC count')
			ax.set_ylabel('Voltage (V)')
			ax.xaxis.set_ticks_position('both')
			ax.yaxis.set_ticks_position('both')
			plt.minorticks_on()
			plt.show()

		return

	def plot_centers(self):

		fig, ax = plt.subplots()
		fig.set_size_inches([10.5,8])
		xbins, ybins = np.linspace(0,200,201), np.linspace(0,200,201)
		h, xedges, yedges, image_mesh = ax.hist2d(self.hpos, self.vpos, bins=(xbins, ybins))#, norm=matplotlib.colors.LogNorm())
		ax.set_xlabel("dt(pulse, reflection)*v [mm]")
		ax.set_ylabel("Y position (perpendicular to strips) [mm]")
		fig.colorbar(image_mesh, ax=ax)

		fig2, ax2 = plt.subplots()
		fig2.set_size_inches([10.5,8])
		h, xedges, yedges, image_mesh = ax2.hist2d(self.hpos, self.vpos, bins=(xbins, ybins), norm=colors.LogNorm())
		ax2.set_xlabel("dt(pulse, reflection)*v [mm]")
		ax2.set_ylabel("Y position (perpendicular to strips) [mm]")
		fig2.colorbar(image_mesh, ax=ax2)

		fig3, ax3 = plt.subplots()
		ax3.hist(self.hpos, np.linspace(90, 200, 111))
		
		plt.show()

		return

	def save_npz(self, file_name):
		#TODO: when we convert all of this data into a dictionary format, we can use
		#np.savez('npz/' + file_name, **data_dict). Then change the load function to
		#data_dict = np.load('npz/' + file_name)
		#Loop through each key and extract the "item" from each element
		#for key in data_dict:
		#	data_dict[key] = data_dict[key].item()


		np.savez('npz/' + file_name, acdc_id=self.acdc_id, lappd_id=self.lappd_id, acc_id=self.acc_id, station_id=self.station_id, zpos=self.zpos, corner_offset=self.corner_offset, waveforms_optch=self.waveforms_optch, waveforms_sin=self.waveforms_sin, hpos=self.hpos, vpos=self.vpos, times_wr=self.times_wr, eventphi=self.eventphi, first_peak=self.first_peak, phi=self.phi, omega=self.omega, delta_t=self.delta_t, opt_chs=self.opt_chs, chi2=self.chi2, startcap=self.startcap)

		return
	
	def load_npz(self, file_name):

		if '.npz' not in file_name:
			file_name += '.npz'

		with np.load('npz/' + file_name) as data:
			self.acdc_id = data['acdc_id']
			self.lappd_id = data['lappd_id']
			self.acc_id = data['acc_id']
			self.station_id = data['station_id']
			self.zpos = data['zpos']
			self.corner_offset = data['corner_offset']

			#This is populated during preprocessing
			self.waveforms_optch = data['waveforms_optch']
			self.waveforms_sin = data['waveforms_sin']
			self.hpos = data['hpos']
			self.vpos = data['vpos']
			self.times_wr = data['times_wr']
			self.eventphi = data['eventphi']
			self.first_peak = data['first_peak']
			self.phi = data['phi']
			self.omega = data['omega']
			self.delta_t = data['delta_t']
			self.opt_chs = data['opt_chs']
			self.chi2 = data['chi2']
			self.startcap = data['startcap']

		return


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
		print(self.cur_waveforms.shape)
		y_data_list = self.cur_waveforms[event,channels,:].reshape(len(channels), -1)
		y_data_raw_list = self.cur_waveforms_raw[event,channels,:].reshape(len(channels), -1)


		fig, (ax1, ax2) = plt.subplots(2, 1)

		# Plots the raw waveform data
		for channel, y_data_raw in enumerate(y_data_raw_list):
			ax1.plot(np.linspace(0,255,256), y_data_raw, label="Channel %i"%channel)
			# ax1.plot(np.linspace(0,255,256), y_data_raw, label="Channel %i"%channel)

		# Plots the corrected waveform data
		for channel, y_data in enumerate(y_data_list):
			ax2.plot(self.sample_times[event, channel], y_data, label='Channel %i'%channel)	
			# ax2.plot(np.linspace(0,255,256), y_data, label='Channel %i'%channel)		

		print(self.sample_times[event, channel])
		# Labels the plots, make them look pretty, and displays the plots
		ax1.set_xlabel("Sample number")
		ax1.set_ylabel("ADC count (raw)")
		ax1.tick_params(right=True, top=True)
		ax2.set_xlabel("Time sample (ns)")
		ax2.set_ylabel("Y-value (calibrated)")
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

		#TODO: add another layer of abstraction: station level, which does not process raw waveform. -JIN-
		#do math to look at coincidence of clocks. 
		return 0 #or 1, or a list of those that are in coincidence vs those that are not.

