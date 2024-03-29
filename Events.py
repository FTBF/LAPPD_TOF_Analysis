import numpy as np
import pandas as pd
import uproot

#This Event class contains the ACDCs presently active in the system
#and then collects information from each ACDC that corresponds
#to properties of the data that has been analyzed in the ACDC class. 
#it collates that information into a "reduced data" series, which can
#be appended to a larger data frame to be shipped out for post processing
#or is used in live-time display of analyzed data outputs. 
#TODO: For efficiency, event class is created after analysis done in ACDC class.

#Now with root ttree instead of pandas dataframe.

class Events:
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
		self.red_cols = {}
		for station in ["Station0", "Station1", "Station2", "Station3"]:
			self.red_cols[station + " X"] = uproot.newbranch(np.float64) #x reconstructed position
			self.red_cols[station + " Y"] = uproot.newbranch(np.float64) #y reconstructed position
			self.red_cols[station + " dX"] = uproot.newbranch(np.float64) #uncertainty in x position
			self.red_cols[station + " dY"] = uproot.newbranch(np.float64) #uncertainty in y position
			self.red_cols[station + " Theta"] = uproot.newbranch(np.float64) #theta reconstructed angle, two or more positions required
			self.red_cols[station + " dTheta"] = uproot.newbranch(np.float64) #uncertainty in theta
			self.red_cols[station + " Phi"] = uproot.newbranch(np.float64) #phi reconstructed angle, two or more positions required
			self.red_cols[station + " dPhi"] = uproot.newbranch(np.float64) #uncertainty in phi
			self.red_cols[station + " T"] = uproot.newbranch(np.float64) #time of arrival, ps
			self.red_cols[station + " dT"] = uproot.newbranch(np.float64)#uncertainty in t
			self.red_cols[station + " Total Charge"] = uproot.newbranch(np.float64) #sum of integrated channel waveforms
			self.red_cols[station + " Max Pulse Height"] = uproot.newbranch(np.float64) #pulse height of largest channel
			self.red_cols[station + " Channels Hit"] = uproot.newbranch(np.float64)#list of channels that pass thresholds
			self.red_cols[station + " Max Channel"] = uproot.newbranch(np.int32) #which channel has largest pulse

		self.red_cols["Mass"] = uproot.newbranch(np.float64)
		self.red_cols["dMass"] = uproot.newbranch(np.float64)
		self.ttree = uproot.newtree(self.red_cols) #create a tree with the above branches


	def get_reduced_series(self):
		return self.red_df #return the filled reduced data series

	###the below functions only makes sense after feeding the acdc objects with the waveforms!###

	
	# #looking at whether clocks/triggers
	# #align to within "window" nanoseconds
	# def check_coincidence(self, window=30):
	# 	coarse_clocks = {}
	# 	fine_clocks = {}
	# 	for a in self.acdcs:
	# 		coarse_clocks[a.get_lappd_station()] = a.get_coarse_clock()
	# 		fine_clocks[a.get_lappd_station()] = a.get_fine_clock()

	# 	#do math to look at coincidence of clocks. 
	# 	return 0 #or 1, or a list of those that are in coincidence vs those that are not.
	
	# #Correcting the raw waveform #1!
	# #Look up 'pedestal_counts' from acdc.py   
	# def baseline_subtract(self):
	# 	for a in self.acdcs:
	# 		a.baseline_subtract()

	# #Correcting the raw waveform #2!
	# #Look up 'voltage_count_curve' from acdc.py
	# def voltage_linearization(self):
	# 	for a in self.acdcs:
	# 		a.voltage_linearization()

	# def coarse_pulse_detect(self):
	# 	pass#NOT YET IMPLEMENTED
	# def position_reconstruction(self):
	# 	pass#NOT YET IMPLEMENTED
	# def time_reconstruction(self):
	# 	pass#NOT YET IMPLEMENTED
