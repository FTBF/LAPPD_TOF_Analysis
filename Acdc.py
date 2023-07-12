import numpy as np 
import pandas as pd
import bitstruct.c as bitstruct
from matplotlib import pyplot as plt


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
		self.times = init_data_dict['times']				# ps, xxx need better variable name and also better description imo; a list of timebase calibrated times
		self.wraparound = init_data_dict['wraparound']		# ps, shape=(# channels,);  a constant time associated with the delay for when the VCDL goes from the 255th cap to the 0th cap
		self.vel = init_data_dict['vel']					# mm/ps; average (~500 MHz - 1GHz) propagation velocity of the strip
		self.dt = init_data_dict['dt']						# ps; nominal sampling time interval, 1/(clock to PSEC4 x number of samples)
		_, _, self.pedestal_data = self.import_raw_data(init_data_dict['pedestal_data_path'], is_pedestal_data=True)	# ADC count, shape=(# events, # channels, # capacitors)
		self.pedestal_counts = init_data_dict['pedestal_counts']	# ADC count; a list of 256 integers, which corresponds to each capicitors of VCDL.
		self.pedestal_voltage = init_data_dict['pedestal_voltage']
		self.voltage_count_curves = init_data_dict['voltage_count_curves']

		# Imports waveform data if a path was specified upon Acdc initialization
		if raw_waveform_data_path_list is not None:

			# Format of the variables import_raw_data writes to:
			#	self.cur_times: xxx, shape=(# events,)
			# 	self.cur_times_320: xxx, shape=(#events,)
			# 	self.cur_waveforms_raw: # ADC count, shape=(# events, # channels, # capacitors); a list of waveforms (2D arrays) for all events, raw means no processing/pedestal subtraction/voltage conversion
			self.import_raw_data(raw_waveform_data_path_list)
			self.process_raw_data()

		else:

			print('Initializing ACDC object with no waveform data.')
			self.cur_times, self.cur_times_320, self.cur_waveforms_raw = None, None, None




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

	
	def initialize_dataframe(self):
		self.df = pd.DataFrame(columns=self.columns)
		for ch in range(30):
			if(ch == self.sync_ch):
				self.sync_dict = {"ch": self.sync_ch, "waveform": None, "times": None, "wraparound": None}
			else:
				s = pd.Series()
				s["ch"] = ch 
				# commented below out - cameron
				# self.df = self.df.append(s, ignore_index=True) #this initializes an empty row with "ch" as an index, that can be edited later

		#now turn the "ch" column into the indices for this dataframe
		self.df = self.df.set_index("ch")


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
		if not isinstance(raw_data_path_list, list):
			raw_data_path_list = [raw_data_path_list]

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

		# Still returns relevant data
		return times_320, times, data
	
	def process_raw_data(self):
		"""Cleans up raw waveform data. Current implementation simply subtracts of average pedestal ADC count of each capacitor in each channel. Future implementations will use voltage_count_curves and interpolation to correct ADC counts and convert to voltage.
		Arguments:
			(Acdc) self		
		"""

		# Averages pedestal_data over all the events
		self.pedestal_counts = self.pedestal_data.mean(0)

		# Subtracts pedestal_counts from the waveform data for each event (subtracts by broadcasting the 2D pedestal_counts array to the 3D cur_waveforms_raw array)
		self.cur_waveforms = self.cur_waveforms_raw - self.pedestal_counts

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
		'times': None,			# xxx need a better name for this
		'wraparound': None,
		'vel': 0.18,			 # mm/ps, average (~500 MHz - 1GHz) propagation velocity of the strip 
		'dt': 1.0/(40e6*256),	 # picoseconds, nominal sampling time interval, 1/(clock to PSEC4 x number of samples)
		'pedestal_data_path': r'/home/cameronpoe/Desktop/lappd_tof_container/testData/Raw_testData_20230615_164912_b0.txt',
		'pedestal_counts': None,
		'pedestal_voltage': None,
		'voltage_count_curves': None
	}

	data_path = r'/home/cameronpoe/Desktop/lappd_tof_container/testData/Raw_testData_20230615_170611_b0.txt'

	test_acdc = Acdc(init_dict)

	test_acdc.import_raw_data(data_path)

	print(test_acdc.cur_times)
	
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




