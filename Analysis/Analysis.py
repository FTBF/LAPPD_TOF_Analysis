import numpy as np
import pandas as pd
import uproot
import yaml
import awkward as ak #tested with version 2.6.1
#This Event class contains the ACDCs presently active in the system
#and then collects information from each ACDC that corresponds
#to properties of the data that has been analyzed in the ACDC class. 
#it collates that information into a "reduced data" series, which can
#be appended to a larger data frame to be shipped out for post processing
#or is used in live-time display of analyzed data outputs. 
#TODO: For efficiency, event class is created after analysis done in ACDC class.

#Now with root ttree instead of pandas dataframe.

class Analysis:
	def __init__(self, config_data, acdcs):

		self.acdcs = acdcs #class is only passed the acdcs that are selected as active by the GUI.
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

		self.c = config_data
		self.tracks = None #Multistation events are named "tracks". Populated by the track finding algorithm, which in order invokes coincidence algo, time of flight algo, and so on.
		
	def convert_wr_counter_in_seconds(self, acdc):
		#The first 32 bits are PPS. The other 32 bits are 250 MHz counters. We return the number in seconds.
		PPS = acdc.rqs["pps"]
		counter = acdc.rqs["wr_time"]
		return PPS + counter/250e6

	def construct_tracks(self, acdc1, acdc2):
		temp_tracks = []
		#loop over each raw data file
		wr_first_station_seconds = self.convert_wr_counter_in_seconds(acdc1)
		for event_pairs in self.construct_coincidence(acdc1, acdc2)[0]: #This operation can be parallelized in the future.
			track_index = 0
			
			#reformat the times and data and pack into our data structure
			temp = {}
			temp["track_id"] = track_index
			
			temp["wr_seconds"] = wr_first_station_seconds[event_pairs[0]]
			
			temp["stations_acdc_id"] = [acdc1.c["acdc_id"], acdc2.c["acdc_id"]]
			temp["stations_lappd_id"] = [acdc1.c["lappd_id"], acdc2.c["lappd_id"]]
			temp["stations_z"] = [acdc1.c["zpos"], acdc2.c["zpos"]]
			temp["stations_x"] = [acdc1.c["corner_offset"][0], acdc2.c["corner_offset"][0]]
			temp["stations_y"] = [acdc1.c["corner_offset"][1], acdc2.c["corner_offset"][1]]
			temp["stations_orientation"] = ["Id", "Id"] #placeholder for now

			temp["polar_angle_phi"], temp["polar_angle_theta"] = self.construct_particle_direction(acdc1, acdc2, event_pairs) 

			#time of flight calculation, mod 4 nanoseconds.
			temp["time_of_flight_ns"] = [np.remainder(acdc2.events["wr_phi"][event_pairs[1]] - acdc1.events["wr_phi"][event_pairs[0]], 2*np.pi)  * 4 / (2*np.pi)] #nanoseconds, 4 ns per cycle

			temp["filenames"] = [acdc1.events["filename"][event_pairs[0]], acdc2.events["filename"][event_pairs[1]]]
			temp["file_timestamps"] = [acdc1.events["file_timestamp"][event_pairs[0]], acdc2.events["file_timestamp"][event_pairs[1]]]

			temp["involved_evts"] = [event_pairs[0], event_pairs[1]]
			temp_tracks.append(temp)

			track_index += 1
		
		#package into an awkward array
		self.tracks = ak.Array(temp_tracks)

	# Two station version of the particle direction algorithm. For more than two stations, linear regression must be implemented.
	# If the stations are in different orientations, the algorithm must be modified to account for that.
	def construct_particle_direction(self, acdc1, acdc2, event_pairs):
		xy_hitpos1 = np.array([acdc1.c["corner_offset"][0]+acdc1.events["hpos"][event_pairs[0]], acdc1.c["corner_offset"][1]+acdc1.events["vpos"][event_pairs[0]]])
		xy_hitpos2 = np.array([acdc2.c["corner_offset"][0]+acdc2.events["hpos"][event_pairs[1]], acdc2.c["corner_offset"][1]+acdc2.events["vpos"][event_pairs[1]]])
		
		#Basic polar angle calculation, assuming the stations are in the same orientation.
		theta = np.arctan((acdc2.c["zpos"] - acdc1.c["zpos"])/(np.linalg.norm( xy_hitpos2 - xy_hitpos1)))
		phi = np.arctan2(xy_hitpos2[1] - xy_hitpos1[1], xy_hitpos2[0] - xy_hitpos1[0])
		return phi, theta
		
	def construct_coincidence(self, acdc1, acdc2):
		#Scan through all events in acdc1. For each event, check if there is a corresponding event in acdc2. The WR clock count is allowed to be off by a certain amount.
		#Output the list of pairs of indexes of events that are within the WR_COUNT_TOLERANCE of each other.
		WR_COUNT_TOLERANCE = 1e-8 #10 ns
		coincidences = []
		acdc1_wr_seconds = self.convert_wr_counter_in_seconds(acdc1)
		acdc2_wr_seconds = self.convert_wr_counter_in_seconds(acdc2)
		#First, we use the assumption that the second list is displaced from the first list by a constant integer. Our initial guess is the difference of medians.
		median_diff_int_seconds = round(np.median(acdc2_wr_seconds) - np.median(acdc1_wr_seconds))
		j2 = 0 #Dynamic programming to speed up the search
		for i in range(len(acdc1_wr_seconds)):
			for j in range(j2, len(acdc2_wr_seconds)):
				if abs(acdc2_wr_seconds[j] - acdc1_wr_seconds[i]-median_diff_int_seconds) < WR_COUNT_TOLERANCE:
					coincidences.append((i,j, acdc2_wr_seconds[j] - acdc1_wr_seconds[i]-median_diff_int_seconds))
				elif acdc2_wr_seconds[j] - acdc1_wr_seconds[i]-median_diff_int_seconds > WR_COUNT_TOLERANCE:
					j2 = j
					break
		print("construct_coincidence found", len(coincidences), "coincidences.")
		return coincidences, median_diff_int_seconds

	def initialize_rqs(self):
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
		
		self.rqs = {}
		#initialize the global event branches (not channel specific)
		for key, str_type in self.rq_config["global"].items():
			self.rqs[key] = []


	def reduce_data(self):

		############initialization functions##############
		#initialize the reduced quantities data structure
		self.initialize_rqs()

		#populate that RQ structure with some carryover quantities
		#from the self.events array that were just from raw files. 
		for key, str_type in self.rq_config["global"].items():
			self.rqs[key] = self.tracks[key]