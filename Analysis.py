import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import uproot
import yaml

class Analysis:
	def __init__(self, infile_name):
		self.infile_name = infile_name
		self.rq_dict = {}
		self.load_npz(self.infile_name) #populates rq_dict from the file
		

	def load_npz(self, file_name):

		file_name = file_name.strip()
		if file_name[-4:] != '.npz':
			file_name = file_name.split('.')[-2] + '.npz'

		with np.load(file_name) as data:
			for key, val in data.items():
				self.rq_dict[key] = val
	

#this class will load in reduced quantity data from the ACDCs (multiple) 
#and perform math, make plots, common operations that don't live in scripts
#or notebooks. 


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

