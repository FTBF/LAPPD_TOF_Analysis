import numpy as np 
import yaml


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
		self.columns = ["ch", "waveform", "position", "len_cor", "times", "wraparound"]
		self.df = pd.DataFrame(columns=columns)
		#one channel is special, used for synchronization, so we keep it separate 
		self.sync_ch = 0
		self.sync_dict = {"ch": self.sync_ch, "waveform": None, "times": None, "wraparound": None} #similar columns as df, currently hard coding the sync channel as "0"

		#initialize the dataframe, without sync channel
		self.initialize_dataframe()

		#metadata dictionary from the loader of the raw files, holds clock/counter information
		self.metadata = {} #todo: fill in, input from Joe

		#general parameters for use in analysis
		self.vel = 0.18 #mm/ps, average (~500 MHz - 1GHz) propagation velocity of the strip 
		self.dt = 1.0/(40e6*256) #picoseconds, sampling time interval, 1/(clock to PSEC4 x number of samples) 

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

		if(calibration_fn is None):
			print("No configuration file selected on initializing Acdc objects, using default values")
			chs = range(30)
			strip_space = 6.9 #mm
			for ch in chs:
				if(ch == self.sync_ch):
					self.sync_dict["times"] = np.arange(0, 256*self.dt, self.dt)
					self.sync_dict["wraparound"] = 400 #ps
					continue 
				
				#edit the entries of the channel
				self.df.at[ch, "waveform"] = None #load in on event loop 
				self.df.at[ch, "position"] = strip_space*ch
				self.df.at[ch, "len_cor"] = 0
				self.df.at[ch, "times"] = np.arange(0, 256*self.dt, self.dt)
				self.df.at[ch, "wraparound"] = 400 #ps

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
	def update_waveforms(self, waves):
		#a dictionary of waveforms, with channel numbers, includes the sync wave
		for ch in waves:
			if(ch == self.sync_dict["ch"]):
				self.sync_dict["waveform"] = waves[ch]
				continue 

			self.df.at[ch, "waveform"] = waves[ch]


	def baseline_subtract(self):
		pass





