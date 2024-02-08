import numpy as np
import pandas as pd
import Event 
import Acdc
from datetime import datetime
import yaml

from matplotlib import pyplot as plt

from Acdc import Acdc
from Util import Util

C_m_per_s = 2.99792e8

#Assume that the data logging script saves a new timestamped
#file at every FTBF spill, i.e. separating each spill into a new
#file. This Loader class is initialized on each spill_file, parses
#it, and sends the event objects to the analysis routine that calls it. 

#The board_config file is a yaml file that has information about which board IDs 
#are connected to which LAPPD IDs at which stations, with also a Z calibration for that station. 

class Loader:
	def __init__(self, spill_fn, board_config=None):
		self.timestamp = self.get_timestamp_from_fn(spill_fn) #datetime object
		self.fn = spill_fn #the name of the raw data file

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

		for a in self.acdcs:
			raw_times_320, raw_times, raw_events = Util.getDataRaw([self.fn+"_"+str(a.get_acdc_id())])#TODO: specify file name
			a.update_waveforms(raw_events, raw_times_320, raw_times) #update the waveforms for each event

			#at this stage all of the ACDCs have had their waveforms updated.
			#create an event object that holds the ACDCs and tell the event object 
			#which analysis steps to run. 
			e = Event.Event(num=0, acdcs=self.acdcs)#we will get rid of the event class.
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
		#ex) board_id = 0 -> acdc_0.h5
		#    board_id = 1 -> acdc_1.h5
		#    ...
		return "dummy_calibration_file.h5"#NOT YET IMPLEMENTED
		
def import_arrays(file_name):

	events = np.load(file_name)

	return events

def twofold_event_coincidence(events1, events2):

	station_dist = 6e-9

	events1[1] -= int(events1[1][0])
	events2[1] -= int(events2[1][0])

	# fig, ax = plt.subplots()
	# ax.hist(events1[1], bins=np.linspace(26, 28,500))
	# plt.show()

	# exit()

	# fig, ax = plt.subplots()
	# # ax.hist(events1[1], bins=400, label='ACDC52')
	# # ax.hist(events2[1], bins=400, label='ACDC62')
	# # ax.hist(events1[1], bins=np.linspace(57, 60, 400), label='ACDC52')
	# # ax.hist(events2[1], bins=np.linspace(57, 60, 400), label='ACDC62')
	# ax.hist(events1[1], bins=np.linspace(57.97, 57.98, 400), label='ACDC52')
	# ax.hist(events2[1], bins=np.linspace(57.97, 57.98, 400), label='ACDC62')
	# ax.set_xlabel('250MHz time (s)')
	# ax.set_ylabel('# of events')
	# ax.xaxis.set_ticks_position('both')
	# ax.yaxis.set_ticks_position('both')
	# plt.minorticks_on()
	# plt.show()
	
	print(events1[1].shape)
	print(events2[1].shape)

	alltimes = np.append(events1[1], events2[1])
	allids = np.append(events1[0], events2[0])
	timereorder = np.argsort(alltimes)
	alltimes = alltimes[timereorder]
	allids = allids[timereorder]
	same_id_mask = np.diff(allids) != 0
	diffs = np.diff(alltimes)
	print(diffs.shape)
	diffs = diffs[same_id_mask]
	print(diffs.shape)
	print(diffs)

	fig, ax = plt.subplots()
	ax.hist(diffs, bins=np.linspace(0, 1e-7, 200))
	plt.show()

	return

def nfold_coincidence(acdcs_list):
	"""
	Returns a 2d numpy array (shape=[# coincident, # stations]) of event numbers that meet a coincidence threshold for each station.
	"""

	threshold = 25e-9
	
	num_stations = len(acdcs_list)

	allids_times = np.empty((3,0))
	for acdc in acdcs_list:
		ids_times = np.vstack([np.full_like(acdc.times, acdc.station_id), np.linspace(0, acdc.times.shape[0]-1, acdc.times.shape[0], dtype=int), np.copy(acdc.times)])
		ids_times[2] -= (int(ids_times[2].min()) + acdc.zpos/C_m_per_s)
		allids_times = np.hstack([allids_times, ids_times])
		print(acdc.station_id)

	# fig, ax = plt.subplots()
	# # ax.hist(acdcs_list[0].times - int(acdcs_list[0].times.min()) + acdcs_list[0].zpos/C_m_per_s, bins=200)
	# # ax.hist(acdcs_list[1].times - int(acdcs_list[1].times.min()) + 0*acdcs_list[1].zpos/C_m_per_s, bins=200)
	# ax.hist(acdcs_list[0].times - int(acdcs_list[0].times[0]), bins=200)
	# ax.hist(acdcs_list[1].times - int(acdcs_list[1].times[0]), bins=200)
	# plt.show()

	# fig, ax = plt.subplots()
	# bins = np.linspace(58.0003635,58.0003639,100)
	# # ax.hist(allids_times[2,allids_times[0] == 1], bins=bins)
	# # ax.hist(allids_times[2,allids_times[0] == 2], bins=bins)
	# ax.hist([allids_times[2,allids_times[0] == 1], allids_times[2,allids_times[0] == 2]], bins=bins)
	# plt.show()


	allids_times = allids_times[:,allids_times[2,:].argsort()]

	diffs = np.diff(allids_times, axis=1)
	mask = (diffs[0,:] != 0) & (diffs[2,:] < threshold)

	fig, ax = plt.subplots()
	bins = np.linspace(-20e-9, 20e-9, 200)
	# bins=10
	ax.hist(diffs[2,:],bins=bins)
	plt.show()

	mask = np.insert(mask,0,False)
	mask = np.append(mask,False)
	false_indices = np.linspace(0,mask.shape[0]-1, mask.shape[0], dtype=int)[np.invert(mask)]
	starts = np.delete(false_indices, -1)[np.diff(false_indices) == num_stations]
	mask = np.full(mask.shape[0]-1, False)
	mask[starts] = True
	mask = np.append(mask, [False]*(num_stations-1))
	for i in range(num_stations-1):
		mask = mask | np.roll(mask, 1)
		mask = np.delete(mask, -1)

	coincident = allids_times[:,mask]
	coincident = coincident.reshape(3, int(coincident.shape[1]/num_stations), num_stations)

	reorder = (coincident[0]).argsort(axis=1)
	coincident = np.take_along_axis(coincident, reorder[np.newaxis,:,:], axis=2)

	event_nums = np.array(coincident[1], dtype=int)

	print(reorder)

	fig, ax = plt.subplots()
	bins = np.linspace(-20e-9, 20e-9, 200)
	# bins = 200
	ax.hist(coincident[2,:,1] - coincident[2,:,0], bins=bins)
	plt.show()	


	print(f'Coincident events: {event_nums.shape[0]}')

	return event_nums

def twofold_mass_squared(acdc_pair, event_nums, momentum):

	acdc1, acdc2 = acdc_pair
	mask1, mask2 = event_nums[:,0], event_nums[:,1]
	hpos1 = acdc1.hpos[mask1] + acdc1.corner_offset[0]
	hpos2 = acdc2.hpos[mask2] + acdc2.corner_offset[0]
	hdist = (hpos2 - hpos1)/100.		# convert to m

	vpos1 = acdc1.vpos[mask1] + acdc1.corner_offset[1]
	vpos2 = acdc2.vpos[mask2] + acdc2.corner_offset[1]
	vdist = (vpos2 - vpos1)/100.		# convert to m

	zdist = acdc2.zpos - acdc1.zpos

	dist = np.sqrt(hdist**2 + vdist**2 + zdist**2)/C_m_per_s	# dist in ns

	times1 = acdc1.times[mask1] - int(acdc1.times[0])
	times2 = acdc2.times[mask2] - int(acdc2.times[0])
	tof = times2 - times1

	mass_squared = (momentum**2)*(-1*tof/dist - 1)

	fig, ax = plt.subplots()
	# bins = np.linspace(-20e-9, 20e-9, 200)
	bins = 200
	ax.hist(mass_squared, bins=bins)
	plt.show()

	return

if __name__ == '__main__':
	acdc62 = Acdc('acdc62')
	acdc62.load_npz('acdc62')
	acdc52 = Acdc('acdc52')
	acdc52.load_npz('acdc52')
	
	acdcs_list = [acdc52, acdc62]
	event_nums = nfold_coincidence(acdcs_list)

	acdc_pair = [acdc52, acdc62]
	twofold_mass_squared(acdc_pair, event_nums, 120)



	


