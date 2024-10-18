import numpy as np 
import bitstruct.c as bitstruct
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline, CubicSpline, PPoly
from scipy.signal import find_peaks, savgol_filter
import uproot
from time import process_time, time
from multiprocessing import Pool
import warnings
import yaml
import util as util


# Globals for debugging purposes only
all_xh = []
all_yh = []


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
		#of the config_data into something that is more compact
		self.c = config_data

		# Constants
		self.chan_rearrange = np.array([5,4,3,2,1,0,11,10,9,8,7,6,17,16,15,14,13,12,23,22,21,20,19,18,29,28,27,26,25,24])
		self.dt = 1.0/(40e6*256)*1e9	# PLL is 40 MHz, DLL is 256 samples, nominal sample spacing = 1/(256*40 MHz)
				
		self.sync_ch = self.chan_rearrange[self.c['sync_ch']]

		# Pedestal related - this may be removed soon and instead reference the ped setting
		#as well as the linearity calibration data. 
		self.ped_data_path = config_data['pedestal_file_name']
		self.pedestal_data = None
		self.pedestal_counts = None

		# Calibration related
		self.calib_data_file_path = config_data['calib_file_name']
		self.vccs = None


		#the output data structure contains many reduced
		#quantities that form an event indexed dataframe. 
		#load the yaml file containing these reduced quantities
		#notes on reduced data output structure

		try:
			with open(self.c["rq_file"], 'r') as yf:
				self.rq_config = yaml.safe_load(yf)
		except FileNotFoundError:
			print(f'{self.c["rq_file"]} doesn\'t exist')
			self.rq_config = None
		
		self.output = {}
		for key, init_value in self.rq_config.items():
			self.output[key] = init_value

		if not self.c["QUIET"]:
			print(f'ACDC instantiated\n  ACDC:    {self.c["acdc_id"]}\n  LAPPD:   {self.c["lappd_id"]}\n  ACC:     {self.c["acc_id"]}\n  Station: {self.c["station_id"]}\n')



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
		if not self.c["QUIET"]:
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

	#calibrate time base and linearity for the board using the .root calibration files. 
	def calibrate_board(self):

		vccs = [[None]*256 for i in range(30)]
		if self.c["CALIB_ADC"]:
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

			if not self.c["QUIET"]:
				print('ACDC voltage calibrated with VCCs')
		
		else:
			self.import_raw_data(self.ped_data_path, is_pedestal_data=True)

			if not self.c["QUIET"]:
				print('ACDC ADC counts calibrated with pedestal data')

		self.vccs = vccs

		if self.c["CALIB_TIME_BASE"]:
			# Creates a time-base calibrated array of sample times
			with uproot.open(self.calib_data_file_path) as calib_file:
				time_base = np.reshape(calib_file['config_tree']['time_offsets'].array(library='np'), (30,256))
				time_base = time_base[self.chan_rearrange,:]*1e9

			if not self.c["QUIET"]:
				print('ACDC sample times calibrated\n')

		else:
			time_base = np.full(256, self.dt) #+ np.random.normal(loc=0., scale=0.010, size=256)
			time_base[-1] += 0.300
			time_base = np.tile(time_base, 30).reshape(30, 256)
			# time_base[:,-1] -= (np.sum(time_base, axis=1) - 25)	# normalizes cumulative samples to 25 ns
			# if (time_base[:,-1] < 0).any():
			# 	print('Error: wrap-around offset less than 0')
			# 	exit()

			if not self.c["QUIET"]:
				print('ACDC sample times NOT calibrated\n')

		self.time_base = time_base

		return

	def preprocess_data(self, data_raw, times_320):

		# Calibrates the ADC data
		data = np.empty_like(data_raw, dtype=np.float64)
		if self.c["CALIB_ADC"]:
			for ch in range(0,30):
				for cap in range(0,256):
					data[:,ch,cap] = self.vccs[ch][cap](data_raw[:,ch,cap])
		else:
			data = data_raw - self.pedestal_counts

		# Selects optimal y data (voltage) and channels
		if self.c["NO_POSITIONS"]:
			ydata_v, opt_chs, misfire_masks = np.full((times_320.shape[0], 256), 0), np.full(times_320.shape[0], 0), np.full((times_320.shape[0], 256), True)
		else:
			ydata_v, opt_chs, misfire_masks = self.determine_optimal_channel(data) # Here we select one channel to be the optimal channel of this event to be analyzed
		
		strip_pos = self.c["strip_pitch"]*np.linspace(0,29,30)
		xdata_v = np.tile(np.copy(strip_pos), ydata_v.shape[0]).reshape(ydata_v.shape[0], 30)
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

		if self.c["DEBUG"]:
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

		preprocess_dict = {
			"xdata_v" : xdata_v,
			"ydata_v" : ydata_v,
			"xdata_h" : xdata_h,
			"ydata_h" : ydata_h,
			"opt_chs" : opt_chs,
			"misfire_masks" : misfire_masks,
			"xdata_sine" : xdata_sine,
			"ydata_sine" : ydata_sine,
			"trigger_low" : trigger_low
		}
		return preprocess_dict

	def determine_optimal_channel(self, data):
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
			if self.c["CALIB_ADC"]:
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

	def calc_positions(self, preprocess_dict, times):

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
		xdata_sine_cut, ydata_sine_cut = preprocess_dict["xdata_sine"][:,cut], preprocess_dict["ydata_sine"][:,cut]

		# Vectorized p0 for sin fit
		B0 = np.average(preprocess_dict["ydata_sine"], axis=1)
		A0 = np.max(preprocess_dict["ydata_sine"], axis=1) - B0
		omega0 = np.full_like(B0, 2*np.pi*0.25)
		phi0 = np.zeros_like(B0)
		p0_array = np.array([A0, omega0, phi0, B0]).T
		
		skipped = []
		for i, (xv, yv, xh, yh, opt_ch, misfire_mask, xsin, ysin, p0, tl) in enumerate(zip(preprocess_dict["xdata_v"], preprocess_dict["ydata_v"], preprocess_dict["xdata_h"], preprocess_dict["ydata_h"], preprocess_dict["opt_chs"], preprocess_dict["misfire_masks"], xdata_sine_cut, ydata_sine_cut, p0_array, preprocess_dict["trigger_low"])):
			try:

				wraparound_ind = 255-tl
				startcap = tl
				xh, yh = xh[misfire_mask], yh[misfire_mask]	
				
				# Finds spatial position
				if self.c["NO_POSITIONS"]:
					delta_t, lbound, vpos = 0, 0, 0
				else:
					delta_t, lbound, reflect_ind = self.calc_delta_t(xh, yh, offsets)
					if wraparound_ind < reflect_ind:	# if wraparound is between peaks throw out event
						raise Exception("Wraparound Error") 
					mu0 = xv[opt_ch]
					vpos = util.calc_vpos(xv, yv, mu0)
				
				# Excludes wraparound in the sine fit
				if self.c["EXCLUDE_WRAP"]:
					# Keeps samples to left of wraparound
					if wraparound_ind >= 128 and wraparound_ind < sin_rbound:		
						xsin, ysin = xsin[:wraparound_ind-sin_lbound], ysin[:wraparound_ind-sin_lbound]
					# Keeps samples to right of wraparound
					elif wraparound_ind < 128 and wraparound_ind > sin_lbound:		
						xsin, ysin = xsin[wraparound_ind-sin_lbound+1:], ysin[wraparound_ind-sin_lbound+1:]
					# Must fit a full period
					if len(xsin) < 40:
						raise Exception("Length Too Short Error")

				# Gets rid of any misfired caps with super low ADC value
				badcap_cut = ysin > 0.2
				xsin, ysin = xsin[badcap_cut], ysin[badcap_cut]
				
				# Fits sine with variable or constant (250 MHz) frequency
				if self.c["NO_SINES"]:
					popt = [0., 0., 0., 0.]
					pcov = [0.]
				elif self.c["VAR_SINE_FREQ"]:
					param_bounds = ([0.025, 1.4, -3*np.pi, 0.6], [0.4, 1.75, 3*np.pi, 0.9])
					popt, pcov = curve_fit(util.sin_const_back, xsin, ysin, p0=p0, bounds=param_bounds)
				else:
					popt, pcov = curve_fit(util.sin_const_back_250, xsin, ysin, p0=(p0[0], p0[2], p0[3]), bounds=([0.025, -3*np.pi, 0.6], [0.4, 3*np.pi, 0.9]))
					popt = [popt[0], 2*np.pi*0.25, popt[1], popt[2]]
				
				# Temporary fit variables for debugging, since the fit gives omega in rad/ns and phi in rad
				# tempomega = popt[1]/(2*np.pi)*1e3
				# tempphi = (popt[2]%(2*np.pi))/(2*np.pi*0.25)
				# if tempphi>=2: tempphi -= 4.

				# Currently not good method of getting goodness-of-fit
				sinsigma = np.sqrt(np.diag(pcov))
				# r = ysin - util.sin_const_back(xsin, *popt)
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
				print(err)
				skipped.append(i)	
				if self.c["DEBUG"]:
					# fig, ax = plt.subplots()
					# ax.scatter(xh, yh, marker='.', color='black', label='Skipped Raw Data')
					# ax.plot(xh, yh, color='black')
					# ax.set_xlabel('Time (ns)', fontdict=dict(size=14))
					# ax.set_ylabel('Voltage (V)', fontdict=dict(size=14))
					# ax.xaxis.set_ticks_position('both')
					# ax.yaxis.set_ticks_position('both')
					# plt.minorticks_on()
					# plt.show()
					pass
		
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
		reflection_time_offset = np.full(len(delta_t_vec) + num_skipped, self.c["strip_length_ns"])
		hpos_vec = 0.5*self.c["vel"]*(delta_t_vec - np.delete(reflection_time_offset, skipped))

		eventphi_vec = (first_peak_vec - (200./self.c["vel"] - hpos_vec/self.c["vel"])) - phi_vec	# 200 mm = length of LAPPD

		preprocess_dict["opt_chs"] = np.delete(preprocess_dict["opt_chs"], skipped)
		
		startcap_vec = np.array(startcap_vec)

		#names of keys in this dict should match reduced_quantities.yml
		rq_dict = {
			"hpos" : hpos_vec,
			"vpos" : vpos_vec,
			"times_wr" : times_wr_vec,
			"eventphi" : eventphi_vec,
			"first_peak" : first_peak_vec,
			"phi" : phi_vec,
			"omega" : omega_vec,
			"delta_t" : delta_t_vec,
			"opt_chs" : preprocess_dict["opt_chs"],
			"chi2" : chi2_vec,
			"startcap" : startcap_vec,
			"num_skipped" : num_skipped
		} 
		
		return rq_dict
	
	def calc_delta_t(self, xh, yh, offsets, debug=False):
		"""
		Returns the time difference between the two peaks in the waveform.
		"""

		lbound, rbound, reflect_ind = util.leading_edge_bounds(xh, yh)

		lsquares = util.find_lsquares(xh, yh, lbound, rbound, offsets)

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

	def plot_raw_lappd(self, event):

		waveform = self.cur_waveforms[event]

		xdata = np.linspace(0,255,256)
		ydata = np.linspace(0,29,30)

		fig, ax = plt.subplots()

		norm = colors.CenteredNorm()
		ax.pcolormesh(xdata, ydata, waveform, norm=norm, cmap='bwr')

		plt.show()

		return

	def process_single_file(self, file_name):
		times_320, times, data_raw = self.import_raw_data(file_name)
		preprocess_dict = self.preprocess_data(data_raw, times_320)
		rq_dict = self.calc_positions(preprocess_dict, times) 
		return rq_dict

	#TODO: add an argument to this which is a list of pedestal files. 
	#Then, with each data file, look at when the last pedestal calibration was done. 
	#If there is a closer-in-time pedestal file, recalibrate with that. 
	def process_files(self, file_list):

		rq_dict = self.output.copy()
		t1 = process_time()
		t3 = time()
		if self.c["MAX_PROCESSES"] != 1:
			with Pool(self.c["MAX_PROCESSES"]) as p:
				file_list = util.convert_to_list(file_list)
				rv = p.map(self.process_single_file, file_list)

			for single_event_rq_dict in rv:
				for key in single_event_rq_dict.keys():
					if key == "num_skipped":
						rq_dict[key] += single_event_rq_dict[key]
					else:
						rq_dict[key].extend(single_event_rq_dict[key])
				

		else:
			for file_name in file_list:
				single_event_rq_dict = self.process_single_file(file_name)
				for key in single_event_rq_dict.keys():
					if key == "num_skipped":
						rq_dict[key] += single_event_rq_dict[key]
					else:
						rq_dict[key].extend(single_event_rq_dict[key])

		t2 = process_time()
		t4 = time()
		total_skipped = rq_dict["num_skipped"]
		num_successful_events = len(rq_dict["hpos"])

		print(f'Total skipped: {round(100.*total_skipped/(total_skipped + num_successful_events),2)}%')
		print(f'Total events analyzed: {total_skipped + num_successful_events}')
		print(f'Process time duration: {round(t2-t1, 3)} s')
		print(f'Wall clock duration: {round(t4-t3, 3)} s')

		for key, val in rq_dict.items():
			if isinstance(val, list):
				self.output[key] = np.array(val)
			else:
				self.output[key] = val
		return

	def save_npz(self, file_name):
		#TODO: when we convert all of this data into a dictionary format, we can use
		#np.savez('npz/' + file_name, **data_dict). Then change the load function to
		#data_dict = np.load('npz/' + file_name)
		#Loop through each key and extract the "item" from each element
		#for key in data_dict:
		#	data_dict[key] = data_dict[key].item()
		np.savez(file_name, **self.output)

		return
	
	def load_npz(self, file_name):
		file_name = file_name.strip()
		if file_name[-4:] != '.npz':
			file_name = file_name.split('.')[-2] + '.npz'

		with np.load(file_name) as data:
			for key, val in data.items():
				self.output[key] = val
	
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
	#Jin suggests the ACDC class should carry a chain of waveforms rather thn every correction function directly acting on a single waveform variable; the former is much more traceable and debuggable.
	#Strategies:
	#Look at all samples, assuming negative polar pulses, take 75% percentile (or something) of maximum sample values
	#Or, just look at the furthest samples from the trigger and do N nanoseconds of averaging. 
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





#________________ Plotting of waveforms functionality __________________________#


#legacy code
# def plot_events(self, file_list):

# 		for file_name in file_list:
# 			times_320, times, data_raw = self.import_raw_data(file_name)
# 			preprocess_vec = self.preprocess_data(data_raw, times_320)
# 			for i in range(len(times)):
# 				fig, ax = plt.subplots()
# 				for chan in range(0,30):
# 					ax.plot(all_xh[i,chan], all_yh[i,chan])
# 				plt.show()

# 		return

# def plot_vccs(self, ch=-1, cap=-1):

#     if ch < 0:
#         ch = int(30*np.random.rand())
#     if cap < 0:
#         cap = int(256*np.random.rand())

#     with uproot.open(self.calib_data_file_path) as calib_file:

#         # Gets numpy array axes are: channel, cap, voltage increment, and ADC type
#         voltage_counts = np.reshape(calib_file["config_tree"]["voltage_count_curves"].array(library="np"), (30,256,256,2))
#         voltage_counts[:,:,:,0] = voltage_counts[:,:,:,0]*1.2/4096.

#         # Filter the data and make it monotonically increasing
#         # voltage_counts[:,:,:,1] = savgol_filter(voltage_counts[:,:,:,1], 41, 2, axis=2)	
#         # reorder = np.argsort(voltage_counts[:,:,:,0], axis=2)
#         # voltage_counts = np.take_along_axis(voltage_counts, reorder[:,:,:,np.newaxis], axis=2)

#         voltage_counts = voltage_counts[self.chan_rearrange,:,:,:]
#         voltage_counts = voltage_counts[ch,cap,:,:]
#         fig, ax = plt.subplots()
#         ax.scatter(voltage_counts[:,1],voltage_counts[:,0], marker='.', color='black', label=f'Ch: {ch}, cap: {cap}')
#         domain = np.linspace(0, 4096, 500)
#         ax.plot(domain, self.vccs[ch][cap](domain), color='red', label='Fit')
#         # ax.axhline(voltage_counts[60, 0], color='green', label='Fit bounds')
#         # ax.axhline(voltage_counts[196, 0], color='green')
#         ax.legend()
#         ax.set_xlabel('ADC count')
#         ax.set_ylabel('Voltage (V)')
#         ax.xaxis.set_ticks_position('both')
#         ax.yaxis.set_ticks_position('both')
#         plt.minorticks_on()
#         plt.show()

#     return
# def plot_centers(self):

#     fig, ax = plt.subplots()
#     fig.set_size_inches([10.5,8])
#     xbins, ybins = np.linspace(0,200,201), np.linspace(0,200,201)
#     h, xedges, yedges, image_mesh = ax.hist2d(self.hpos, self.vpos, bins=(xbins, ybins))#, norm=matplotlib.colors.LogNorm())
#     ax.set_xlabel("dt(pulse, reflection)*v [mm]")
#     ax.set_ylabel("Y position (perpendicular to strips) [mm]")
#     fig.colorbar(image_mesh, ax=ax)

#     fig2, ax2 = plt.subplots()
#     fig2.set_size_inches([10.5,8])
#     h, xedges, yedges, image_mesh = ax2.hist2d(self.hpos, self.vpos, bins=(xbins, ybins), norm=colors.LogNorm())
#     ax2.set_xlabel("dt(pulse, reflection)*v [mm]")
#     ax2.set_ylabel("Y position (perpendicular to strips) [mm]")
#     fig2.colorbar(image_mesh, ax=ax2)

#     fig3, ax3 = plt.subplots()
#     ax3.hist(self.hpos, np.linspace(90, 200, 111))
	
#     plt.show()

#     return


# # not sure how this one is affected by incorporating sample_times and wrap-around fix xxx 
# def hist_single_cap_counts_vs_ped(self, ch, cap):
#     """Plots a histogram of ADC counts for a single capacitor in a channel for all events recorded in the binary file. Also plots a histogram for the pedestal ADC counts of the same capacitor.
#     Arguments:
#         (int): channel number
#         (int): capacitor number		
#     """

#     # Calculates the bins for the histogram using the maximum and minimum ADC counts
#     single_cap_ped_counts = self.pedestal_data[:, ch, cap]
#     ped_bins_left_edge = single_cap_ped_counts.min()
#     ped_bins_right_edge = single_cap_ped_counts.max()+1
#     ped_bins = np.linspace(ped_bins_left_edge, ped_bins_right_edge, ped_bins_right_edge-ped_bins_left_edge+1)
#     print(f'Minimum pedestal ADC count: {ped_bins_left_edge}')
#     print(f'Maximum pedestal ADC count: {ped_bins_right_edge-1}')

#     # Calculates the bins for the histogram using the maximum and minimum ADC counts
#     single_cap_raw_counts = self.cur_waveforms_raw[:, ch, cap]
#     raw_bins_left_edge = single_cap_raw_counts.min()
#     raw_bins_right_edge = single_cap_raw_counts.max()+1
#     raw_bins = np.linspace(raw_bins_left_edge, raw_bins_right_edge, raw_bins_right_edge-raw_bins_left_edge+1)
#     print(f'Minimum raw waveform ADC count: {raw_bins_left_edge}')
#     print(f'Maximum raw waveform ADC count: {raw_bins_right_edge-1}')

#     # Plots histogram
#     fig, ax = plt.subplots()
#     ax.hist(single_cap_ped_counts, histtype='step', linewidth=3, bins=ped_bins)
#     ax.hist(single_cap_raw_counts, histtype='step', linewidth=3, bins=raw_bins)
#     ax.set_xlabel('ADC Counts')
#     ax.set_ylabel('Number of events (per 1 count bins)')
#     ax.set_yscale('log')
#     plt.show()

#     return

# def plot_ped_corrected_pulse(self, event, channels=None):
#     """Plots a single event across multiple channels to compare raw and pedestal-corrected ADC counts.
#     Arguments:
#         (Acdc) self
#         (int) event: the index number of the event you wish to plot
#         (int / list) channels: a single channel or list of channels you wish to plot for the event
#     """

#     # Checks if user specifies a subset of channels, if so, makes sure subset is of type list, if not, uses all channels.
#     if channels is None:
#         channels = np.linspace(0, 29, 30, dtype=int)
#     channels = util.convert_to_list(channels)

#     # Creates 1D array of x_data (all 256 capacitors) and computes 2D array (one axis channel #, other axis capacitor #) of
#     #	corrected and raw ADC data
#     print(self.cur_waveforms.shape)
#     y_data_list = self.cur_waveforms[event,channels,:].reshape(len(channels), -1)
#     y_data_raw_list = self.cur_waveforms_raw[event,channels,:].reshape(len(channels), -1)


#     fig, (ax1, ax2) = plt.subplots(2, 1)

#     # Plots the raw waveform data
#     for channel, y_data_raw in enumerate(y_data_raw_list):
#         ax1.plot(np.linspace(0,255,256), y_data_raw, label="Channel %i"%channel)
#         # ax1.plot(np.linspace(0,255,256), y_data_raw, label="Channel %i"%channel)

#     # Plots the corrected waveform data
#     for channel, y_data in enumerate(y_data_list):
#         ax2.plot(self.sample_times[event, channel], y_data, label='Channel %i'%channel)	
#         # ax2.plot(np.linspace(0,255,256), y_data, label='Channel %i'%channel)		

#     print(self.sample_times[event, channel])
#     # Labels the plots, make them look pretty, and displays the plots
#     ax1.set_xlabel("Sample number")
#     ax1.set_ylabel("ADC count (raw)")
#     ax1.tick_params(right=True, top=True)
#     ax2.set_xlabel("Time sample (ns)")
#     ax2.set_ylabel("Y-value (calibrated)")
#     ax2.tick_params(right=True, top=True)
	
#     fig.tight_layout()
#     plt.show()
#     return
