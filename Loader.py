import numpy as np 
import Event 
import Acdc
from datetime import datetime
import yaml

#Assume that the data logging script saves a new timestamped
#file at every FTBF spill, i.e. separating each spill into a new
#file. This Loader class is initialized on each spill_file, parses
#it, and sends the event objects to the analysis routine that calls it. 

#The board_config file is a yaml file that has information about which board IDs 
#are connected to which LAPPD IDs at which stations, with also a Z calibration for that station. 

class Loader:
	def __init__(self, spill_fn, board_config=None):
		self.timestamp = self.get_timestamp_from_fn(spill_fn) #datetime object
		self.fn = spill_fn

		self.board_config_dict = self.load_config(board_config)

		self.events = [] #list of events to be populated
		self.acdcs = [] #list of ACDC objects specified as active. 


	def load_config(self, fn):
		if(fn is None):
			print("No ACDC/LAPPD board config provided, loading default configuration")
			self.load_config("configs/example_board_config.yml")
			return

		with open(fn, "r") as stream:
			try:
				return yaml.safe_load(stream)
			except Exception as exc:
				print("Had an exception while reading yaml file for board_config in loader class")
				print(exc)

	def initialize_boards(self):
		if(self.board_config_dict is None):
			print("Please load board config dict before initializing ACDC objects in loader class")
			return


		for station, conf in self.board_config_dict.items():
			#if it is not selected to be active, don't create the object
			if(conf["active"] == 0):
				continue

			#get calibration filename for this board
			cal_file = self.get_acdc_calib_file(conf["acdc_id"])
			self.acdcs.append(Acdc.Acdc(conf["acdc_id"], station, conf["lappd_id"], cal_file)) #create ADCD object and save for later. 

		print("Finished initializing empty, but calibrated, ACDC objects")



	def analyze_spill(self):
		#####insert code for parsing of data files#####

		spill_df = pd.DataFrame() #reduced dataframe object, culmination of all event analysis
		#psuedo code for how to load and loop through events
		for i, raw_event in enumerate(raw_events):
			for raw_acdc in raw_event:
				for a in self.adcds:
					#check if the data in the raw event is an actively
					#selected acdc/LAPPD station
					if(a.get_acdc_id() == raw_acdc["id"]):
						raw_waves = raw_acdc["waves"] #a dictionary of waves, raw_waves[ch] = np.array(waveform samples)
						raw_metadata = raw_acdc["metadata"] #also can be a dictionary of metadata variables, like clock cycles
						a.update_waveforms(raw_waves)
						a.update_metadata(raw_metadata)

			#at this stage all of the ACDCs have had their waveforms updated.
			#create an event object that holds the ACDCs and tell the event object 
			#which analysis steps to run. 
			e = Event.Event(num=i, acdcs=self.acdcs)
			coinc = e.check_coincidence(window=30) #check metadata variables to see if there is coincidence between detectors
			if(coinc == False):
				continue

			e.baseline_subtract()
			e.coarse_pulse_detect()
			e.position_reconstruction()
			e.time_reconstruction()

			#clean up and save data
			red_ser = e.get_reduced_series()
			spill_df = spill_df.append(red_ser, ignore_index=True)

			#save it if you want
			#spill_df.to_hdf("output_file.h5")




	def get_acdc_calib_file(self, board_id):
		#this function should have info on where to find these calibration
		#files, and returns the calibration file name
		return "dummy_calibration_file.h5"
		



