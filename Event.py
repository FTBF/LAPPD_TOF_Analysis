import numpy as np
import pandas as pd 

#This Event class contains the ACDCs presently active in the system
#and then collects information from each ACDC that corresponds
#to properties of the data that has been analyzed in the ACDC class. 
#it collates that information into a "reduced data" series, which can
#be appended to a larger data frame to be shipped out for post processing
#or is used in live-time display of analyzed data outputs. 
#TODO: For efficiency, event class is created after analysis done in ACDC class.

class Event:
	def __init__(self, num = None, acdcs=None):

		self.acdcs = acdcs #class is only passed the acdcs that are selected as active by the GUI. 
		self.num = num #event number, identifying it within a larger dataset or spill

		#the output of analyses, a reduced data series, is meant
		#for easy access of simple quantities that come from
		#analyzing raw data. The reduced data series from a single
		#event can be appended to a a dataframe to form a many-event
		#reduced dataset that can be shipped out to users in .h5 hdf5 format.
		#or, it can be stream-connected to the live display and interface. 

		#example reduced data quantities
		self.red_cols = []
		for station in ["Station0", "Station1", "Station2", "Station3"]:
			self.red_cols.append(station + " X") #x reconstructed position
			self.red_cols.append(station + " Y") #y reconstructed position
			self.red_cols.append(station + " dX") #uncertainty in x position
			self.red_cols.append(station + " dY") #uncertainty in y position
			self.red_cols.append(station + " T") #time of arrival, ps
			self.red_cols.append(station + " dT") #uncertainty in t
			self.red_cols.append(station + " Total Charge") #sum of integrated channel waveforms
			self.red_cols.append(station + " Max Pulse Height") #pulse height of largest channel
			self.red_cols.append(station + " Channels Hit") #list of channels that pass thresholds
			self.red_cols.append(station + " Max Channel") #which channel has largest pulse

		
		self.red_df = pd.Series(columns=self.red_cols)


	def get_reduced_series(self):
		return self.red_df #return the filled reduced data series

	###the below functions only makes sense after feeding the acdc objects with the waveforms!###

	
	#looking at whether clocks/triggers
	#align to within "window" nanoseconds
	def check_coincidence(self, window=30):
		coarse_clocks = {}
		fine_clocks = {}
		for a in self.acdcs:
			coarse_clocks[a.get_lappd_station()] = a.get_coarse_clock()
			fine_clocks[a.get_lappd_station()] = a.get_fine_clock()

		#do math to look at coincidence of clocks. 
		return 0 #or 1, or a list of those that are in coincidence vs those that are not.
	
	#Correcting the raw waveform #1!
	#Look up 'pedestal_counts' from acdc.py   
	def baseline_subtract(self):
		for a in self.acdcs:
			a.baseline_subtract()

	#Correcting the raw waveform #2!
	#Look up 'voltage_count_curve' from acdc.py
	def voltage_linearization(self):
		for a in self.acdcs:
			a.voltage_linearization()

	def coarse_pulse_detect(self):
		pass#NOT YET IMPLEMENTED
	def position_reconstruction(self):
		pass#NOT YET IMPLEMENTED
	def time_reconstruction(self):
		pass#NOT YET IMPLEMENTED
