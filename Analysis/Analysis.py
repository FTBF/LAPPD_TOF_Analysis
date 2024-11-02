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

class Analysis:
	def __init__(self, config, acdcs):

		self.acdcs = acdcs #class is only passed the acdcs that are selected as active by the GUI. 
	def convert_wr_counter_in_seconds(self, b64num):
		#The first 32 bits are PPS. The other 32 bits are 250 MHz counters. We return the number in seconds.
		PPS = b64num >> 32
		counter = b64num & 0xFFFFFFFF
		return PPS + counter/250e6
		
	def construct_coincidence(self, acdc1, acdc2):
		#Scan through all events in acdc1. For each event, check if there is a corresponding event in acdc2. The WR clock count is allowed to be off by a certain amount.
		#Output the list of pairs of indexes of events that are within the WR_COUNT_TOLERANCE of each other.
		WR_COUNT_TOLERANCE = 1e-8 #10 ns
		coincidences = []
		acdc1_wr_seconds = self.convert_wr_counter_in_seconds(acdc1.events["wr_time"])
		acdc2_wr_seconds = self.convert_wr_counter_in_seconds(acdc2.events["wr_time"])
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


