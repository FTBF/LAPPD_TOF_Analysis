import numpy as np 
import pandas as pd


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
	def __init__(self, acdc_id=None, lappd_station=None, lappd_id=None, calibration_fn=None, wave_dict=None):

		#identifying variables, passed in from another class that knows about 
		#these for the entire TOF setup. 
		self.acdc_id = acdc_id #ACDC number, like "46" - see the inventory
		self.lappd_station = lappd_station #LAPPD station position, like 0,1,2,3, in the TOF setup as a whole
		self.lappd_id = lappd_id #LAPPD ID like "125" from Incom numbering



		#"ch": channel number, preferentially matches ACDC data output channel number please. (0, 1, ...)
		#"waveform": adc counts, numpy array representing the waveform samples
		#"position": mm, local center positions of the strip relative to the bottom of the LAPPD
		#"len_cor": mm, a correction on the length of the strip + PCB traces in case of large differences. 

		#the ACDCs do not have a constant sampling rate, like in self.dt. Instead, each sample
		#has its time relative to the last sample calibrated and stored in a calibration file. 
		#"times": ps, a list of timebase calibrated times
		#"wraparound": ps, a constant time associated with the delay for when the VCDL goes from the 255th cap to the 0th cap
		

		#PLEASE VALIDATE!? -JIN- Each capacitor of the VCDL will carry slightly different number of charges even when we feed the entire ring buffer with a 0.0v DC signal.
		#As a result, systemic(non-random) fluctuation is visible at the each sample of raw waveforms. We say that each capacitor has a characteristic 'pedestal' ADC count, which stays effectively constant during an entire analysis.
		#baseline_subtract() function removes the formentioned systemic error from the current waveform by subtracting each pedestal ADC counts from the corresponding samples. 
		#"pedestal_counts": ADC count, a list of 256 integers, which corresponds to each capicitors of VCDL.

		#DO WE NEED 'pedestal_counts'? ISN'T 'voltage_count_curve' A SUPERSET OF 'pedestal_counts'? -JIN-
		#ADC counts do not exactly 'measure' the input voltage, in the sense that each capacitor of the VCDL does not charge completely linearly with the input voltage.
		#Thus the 'voltage-ADC count' curve is measured for each capacitor, and we also consider this as a characteristic curve of the capacitor.
		#voltage_linearization() function reconstructs actual voltage waveform from ADC count waveform utilizing inverse function theorem(??? -JIN)
		#"voltage_count_curves": 256(# of capacitors)*[[voltage, ADC count]*(# of measurement points)], # of measurement points typically being 256.
		self.columns = ["ch", "waveform", "position", "len_cor", "times", "wraparound", "pedestal_counts", "voltage_count_curves"]
		self.df = pd.DataFrame(columns=self.columns)
		#one channel is special, used for synchronization, so we keep it separate 
		self.sync_ch = 0
		self.sync_dict = {"ch": self.sync_ch, "waveform": None, "times": None, "wraparound": None} #similar columns as df, currently hard coding the sync channel as "0"

		#initialize the dataframe, without sync channel
		self.initialize_dataframe()

		#metadata dictionary from the loader of the raw files, holds clock/counter information
		self.cur_times = 0 #s, timestamp of the currently loaded waveform
		self.cur_times_320 = 0 #clock cycles, timestamp of the currently loaded waveform
		self.cur_event_count = 0 #event count of the currently loaded waveform
		self.event_numbers = [] #list of event numbers, in order, for the currently loaded waveform.

		#general parameters for use in analysis
		self.vel = 0.18 #mm/ps, average (~500 MHz - 1GHz) propagation velocity of the strip 
		self.dt = 1.0/(40e6*256) #picoseconds, nominal sampling time interval, 1/(clock to PSEC4 x number of samples)
		


		if(wave_dict is None):
			print("Initializing an empty Acdc object with no waveforms")

		self.calibration_fn = calibration_fn #location of config file
		self.load_calibration() #will load default values if no calibration file provided. clear indicates we want a fresh dataframe

	
	def initialize_dataframe(self):
		self.df = pd.DataFrame(columns=self.columns)
		for ch in range(30):
			if(ch == self.sync_ch):
				self.sync_dict = {"ch": self.sync_ch, "waveform": None, "times": None, "wraparound": None}
			else:
				s = pd.Series()
				s["ch"] = ch 
				self.df = self.df.append(s, ignore_index=True) #this initializes an empty row with "ch" as an index, that can be edited later

		#now turn the "ch" column into the indices for this dataframe
		self.df = self.df.set_index("ch")


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





