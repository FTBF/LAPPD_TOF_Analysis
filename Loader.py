import numpy as np
import pandas as pd
import Event 
import Acdc
from datetime import datetime
import yaml
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters

from matplotlib import pyplot as plt
from matplotlib import colors

from Acdc import Acdc
from Util import Util

C_m_per_s = 299792458

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
		
def omega_hist(acdc):

	fig, ax = plt.subplots()
	ax.hist(1e3*acdc.omega/(2*np.pi), bins=np.linspace(235,255,201))
	ax.set_title(f'ACDC {acdc.acdc_id}', fontdict=dict(size=15))
	ax.set_xlabel('Frequency (MHz)', fontdict=dict(size=14))
	ax.set_ylabel('# Events (per 100 KHz bin)', fontdict=dict(size=14))
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.minorticks_on()
	plt.show()

	return

def nfold_coincidence(acdcs_list):
	"""
	Returns a 2d numpy array (shape=[# coincident, # stations]) of event numbers that meet a coincidence threshold for each station.
	"""

	threshold = 25e-9
	
	num_stations = len(acdcs_list)

	allids_times = np.empty((3,0))
	# acdcs_list[1].zpos = 0
	for acdc in acdcs_list:
		num_events = acdc.times_wr.shape[0]
		ids_times = np.vstack([np.full(num_events, acdc.station_id), np.linspace(0, num_events-1, num_events, dtype=int), np.copy(acdc.times_wr)])
		ids_times[2] -= (int(ids_times[2][0]) + acdc.zpos/C_m_per_s)
		ids_times[2] -= (int(ids_times[2][0]))
		allids_times = np.hstack([allids_times, ids_times])
		print(f'Station: {acdc.station_id}')
		print(f'ACDC: {acdc.acdc_id}')
		print(f'zpos: {acdc.zpos} m\n')

	allids_times = allids_times[:,allids_times[2,:].argsort()]

	diffs = np.diff(allids_times, axis=1)

	fig, ax = plt.subplots()
	bins = np.linspace(-20, 20, 200)
	ax.hist(1e9*diffs[2,:],bins=bins)
	ax.set_xlabel('WR time differences (ns)', fontdict=dict(size=14))
	ax.set_ylabel('Events per 200 fs bin', fontdict=dict(size=14))
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.minorticks_on()
	plt.show()

	mask = (diffs[0,:] != 0) & (diffs[2,:] < threshold)
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

	print(coincident)
	print(coincident[0][0][0])
	print(coincident[0][0][1])

	print(coincident[0][20][0])
	print(coincident[0][20][1])


	fig, ax = plt.subplots()
	ax.hist(coincident[0,:,0])
	plt.show()

	reorder = (coincident[0]).argsort(axis=1)
	print(reorder)
	coincident = np.take_along_axis(coincident, reorder[np.newaxis,:,:], axis=2)
	print(coincident[2][0][0])
	print(coincident[2][0][1])

	print(coincident[2][20][0])
	print(coincident[2][20][1])

	event_nums = np.array(coincident[1], dtype=int)

	# fig, ax = plt.subplots()
	# # bins = np.linspace(-10e-9, 20e-9, 300)
	# bins = 200
	# # ax.hist(coincident[2,:,1] - coincident[2,:,0], bins=bins)
	# wrtimes_stat1 = (acdcs_list[0].times_wr - acdcs_list[0].times_wr[0])[event_nums[:,0]]
	# wrtimes_stat2 = (acdcs_list[1].times_wr - acdcs_list[1].times_wr[0])[event_nums[:,1]]
	# # ax.hist(wrtimes_stat1, bins=bins)
	# # ax.hist(wrtimes_stat2, bins=bins)
	# ax.hist(wrtimes_stat2 - wrtimes_stat1, bins=bins)
	# ax.set_ylabel('# Events (per 100 ps bin)', fontdict=dict(size=14))
	# ax.set_xlabel('TOF (ns)', fontdict=dict(size=14))
	# ax.xaxis.set_ticks_position('both')
	# ax.yaxis.set_ticks_position('both')
	# plt.minorticks_on()
	# plt.show()	


	print(f'Coincident events: {event_nums.shape[0]}')

	return event_nums

def twofold_mass_squared(acdc_pair, event_nums, momentum):

	acdc1, acdc2 = acdc_pair
	mask1, mask2 = event_nums[:,0], event_nums[:,1]
	hpos1 = acdc1.hpos[mask1] + acdc1.corner_offset[0]
	hpos2 = acdc2.hpos[mask2] + acdc2.corner_offset[0]
	hdist = (hpos2 - hpos1)/1000.		# convert to m

	vpos1 = acdc1.vpos[mask1] + acdc1.corner_offset[1]
	vpos2 = acdc2.vpos[mask2] + acdc2.corner_offset[1]
	vdist = (vpos2 - vpos1)/1000.		# convert to m

	zdist = acdc2.zpos - acdc1.zpos

	dist = np.sqrt(hdist**2 + vdist**2 + zdist**2)/C_m_per_s	# dist in s

	times1 = acdc1.times[mask1] - int(acdc1.times[0])
	times2 = acdc2.times[mask2] - int(acdc2.times[0])
	tof = times2 - times1

	mass_squared = (momentum**2)*(tof/dist - 1)

	fig, ax = plt.subplots()
	bins = np.linspace(-20e-9, 20e-9, 200)
	bins = 200
	ax.hist(tof, bins=bins)
	plt.show()

	return

def self_calibrate(acdc_pair, event_nums):

	PROTON_MASS = 0.93827208816
	PROTON_MOMENTUM = 120

	acdc_order = np.array([acdc_pair[0].station_id, acdc_pair[1].station_id]).argsort()

	acdc1, acdc2 = acdc_pair[acdc_order[0]], acdc_pair[acdc_order[1]]
	mask1, mask2 = event_nums[:,0], event_nums[:,1]
	hpos1 = acdc1.hpos[mask1] + acdc1.corner_offset[0]
	hpos2 = acdc2.hpos[mask2] + acdc2.corner_offset[0]
	hdist = (hpos2 - hpos1)/1000.		# convert to m
	vpos1 = acdc1.vpos[mask1] + acdc1.corner_offset[1]
	vpos2 = acdc2.vpos[mask2] + acdc2.corner_offset[1]
	vdist = (vpos2 - vpos1)/1000.		# convert to m
	zdist = acdc2.zpos - acdc1.zpos
	dist = np.sqrt(hdist**2 + vdist**2 + zdist**2)/C_m_per_s	# dist in s
	expected_tofs = dist*np.sqrt(1+(PROTON_MASS**2)/(PROTON_MOMENTUM**2))

	optchs1, optchs2 = acdc1.opt_chs[mask1], acdc2.opt_chs[mask2]

	calibconstants = np.empty((30,30), dtype=np.float64)

	wr1_full, wr2_full = acdc1.times_wr, acdc2.times_wr
	wr1_full -= int(wr1_full[0])
	wr2_full -= int(wr2_full[0])

	wr1_cut = wr1_full[mask1]
	wr2_cut = wr2_full[mask2]

	fig, ax = plt.subplots()
	bins = np.linspace(-20e-9, 20e-9, 200)
	ax.hist((wr2_cut-acdc2.zpos/C_m_per_s)-(wr1_cut-acdc1.zpos/C_m_per_s), bins=bins)
	plt.show()

	for i in range(30):
		for j in range(30):
			submask = np.linspace(0, expected_tofs.shape[0]-1, expected_tofs.shape[0], dtype=int)[(optchs1 == i) & (optchs2 == j)]
			if submask.shape[0] < 10:
				continue
			submask1, submask2 = mask1[submask], mask2[submask]
			wr1, wr2 = wr1_full[submask1], wr2_full[submask2]
			eventphi1, eventphi2 = acdc1.eventphi[submask], acdc2.eventphi[submask]
			# xsin1, xsin2 = 1e-9*acdc1.xsin_last[submask1], 1e-9*acdc2.xsin_last[submask2]
			first_peak1, first_peak2 = 1e-9*acdc1.first_peak[submask1], 1e-9*acdc2.first_peak[submask2]
			phi1, phi2 = 1e-9*acdc1.phi[submask1], 1e-9*acdc2.phi[submask2]
			chi1, chi2 = acdc1.chi2[submask1], acdc2.chi2[submask2]
			dt1, dt2 = 1e-9*acdc1.delta_t[submask1], 1e-9*acdc2.delta_t[submask2]
			x = np.array([wr1, wr2, first_peak1, first_peak2, phi1, phi2, dt1, dt2]).T
			y = expected_tofs[submask]%4.
			# reg = LinearRegression().fit(x,y)

			x = (eventphi2 - eventphi1)%4.
			x = x.reshape(-1,1)
			reg = LinearRegression().fit(x,y)

			# fig, ax = plt.subplots()
			# ax.hist(expected_tofs, bins=200)
			# plt.show()


			# print('\n')
			# print(wr1)
			# print(wr2)
			# print(xsin1)
			# print(first_peak1)
			# print(phi1)
			# print(dt1)
			# print(y)
			if x.shape[0] > 1000:
				# fig, ax = plt.subplots()
				# ax.hist(1e9*(phi2-phi1), bins=200)
				# plt.show()
				# for k in range(30):
				# 	fig, ax = plt.subplots()
				# 	ax.hist(1e9*phi1[k], bins=200)
				# 	ax.hist(1e9*phi2[k], bins=200)
				# 	plt.show()

				print('\n')
				print(f'{i}, {j}')
				print(reg.score(x,y))
				print(reg.coef_)
				print(reg.intercept_)

				# fig, ax = plt.subplots()
				# ax.hist(chi2[:,2], bins=200)

				fig, ax = plt.subplots()
				plotting_list = [eventphi1, eventphi2, (eventphi2-eventphi1)]
				# for item in plotting_list:
				# 	histvals, binedges = np.histogram(item, bins=200)
				# 	ax.hist(item, bins=200)
				# ax.hist(acdc1.eventphi[acdc1.opt_chs == i], bins=200)
				# ax.hist(acdc2.eventphi[acdc2.opt_chs == j], bins=200)
				ax.hist(eventphi1, bins=200)
				ax.hist(eventphi2, bins=200)
				ax.hist(eventphi2-eventphi1, bins=200)
				ax.set_xlabel('Phase (ns)', fontdict=dict(size=14))
				ax.set_ylabel('Events', fontdict=dict(size=14))
				ax.xaxis.set_ticks_position('both')
				ax.yaxis.set_ticks_position('both')
				plt.minorticks_on()
				# ax.hist(x,bins=200)
				fig2, ax2 = plt.subplots()
				ax2.hist(wr2-wr1, bins=200)
				plt.show()

				

	return

def plot_phi_sin(acdc, mask=None):

	phis = acdc.phi
	if mask is not None:
		phis = phis[mask]

	fig, ax = plt.subplots()
	ax.hist(phis, bins=np.linspace(-2,2,401), label=f'ACDC {acdc.acdc_id}')
	ax.set_xlabel('250 MHz sine phase (ns)', fontdict=dict(size=14))
	ax.set_ylabel('Events per 10 ps bin', fontdict=dict(size=14))
	ax.legend()
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.minorticks_on()
	plt.show()

	return

def plot_omega_sin(acdc, mask=None):
	
	omegas = 1e3*acdc.omega/(2*np.pi)
	if mask is not None:
		omegas = omegas[mask]

	fig, ax = plt.subplots()
	ax.hist(omegas, bins=np.linspace(237.5, 252.5, 301), label=f'ACDC {acdc.acdc_id}')
	ax.set_xlabel('WR sine frequency (MHz)', fontdict=dict(size=14))
	ax.set_ylabel('Events per 50 kHz bin', fontdict=dict(size=14))
	ax.legend()
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.minorticks_on()
	plt.show()

	return

def plot_omega_phi_sin(acdc, mask=None):

	fig, ax = plt.subplots()
	ax.hist(acdc.startcap,bins=np.linspace(0, 256, 65), label=f'ACDC {acdc.acdc_id}')
	ax.set_xlabel('Trigger_low position (index)', fontdict=dict(size=14))
	ax.set_ylabel('Number of events', fontdict=dict(size=14))
	ax.legend()
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.minorticks_on()
	plt.show()

	phis = acdc.phi
	omegas = 1e3*acdc.omega/(2*np.pi)

	hist2dbins = (np.linspace(217.5, 232.5, 301), np.linspace(-2,2,401))
	hist2dbins = (np.linspace(237.5, 252.5, 301), np.linspace(-2,2,401))
	# hist2dbins = 100

	fig, ax = plt.subplots()
	_, _, _, image_mesh = ax.hist2d(omegas, phis, bins=hist2dbins, label=f'ACDC {acdc.acdc_id}')
	fig.colorbar(image_mesh, ax=ax)
	ax.set_xlabel('WR sine frequency (MHz)', fontdict=dict(size=14))
	ax.set_ylabel('250 MHz sine phase (ns)', fontdict=dict(size=14))
	# ax.legend()
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.minorticks_on()
	plt.show()

	if mask is not None:
		phis = phis[mask]
		omegas = omegas[mask]

	for cap in np.unique(acdc.startcap):
		capmask = acdc.startcap == cap
		fig, ax = plt.subplots()
		ax.set_title(f'Trigger cap {cap}')
		_, _, _, image_mesh = ax.hist2d(omegas[capmask], phis[capmask], bins=hist2dbins, label=f'ACDC {acdc.acdc_id} cw_low: {cap}')
		fig.colorbar(image_mesh, ax=ax)
		ax.set_xlabel('WR sine frequency (MHz)', fontdict=dict(size=14))
		ax.set_ylabel('250 MHz sine phase (ns)', fontdict=dict(size=14))
		# ax.legend()
		ax.xaxis.set_ticks_position('both')
		ax.yaxis.set_ticks_position('both')
		plt.minorticks_on()
		plt.show()

	fig, ax = plt.subplots()
	cut = np.linspace(0, 9999, 10000, dtype=int)
	cut = np.linspace(0,9998, 9999, dtype=int)
	times = acdc.times_wr
	times -= times[0]
	times = np.linspace(0,9999, 10000, dtype=int)
	print(omegas.shape)
	ax.plot(times[cut], omegas[cut], label=f'ACDC {acdc.acdc_id}')
	ax.legend()
	ax.set_xlabel('Event number', fontdict=dict(size=14))
	ax.set_ylabel('Frequency (MHz)', fontdict=dict(size=14))
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.minorticks_on()
	plt.show()

	return

def plot_sin_time_hists(acdc):

	omegas = acdc.omega
	omegas = 1e3*acdc.omega/(2*np.pi)
	times = acdc.times_wr
	times -= times[0]
	lower = 58.5
	upper = 58.6
	omegas = omegas[(times > lower) & (times < upper)]
	times = times[(times > lower) & (times < upper)]

	times_under = times[omegas < 248.5]
	times_over = times[omegas > 248.5]

	fig, ax = plt.subplots()
	ax.hist([times_under, times_over], bins=500, histtype='barstacked', label=['$\omega < 248.5$', '$\omega > 248.5$'])
	# ax.hist(times_over, bins=500, histtype='barstacked')
	ax.legend()
	ax.set_xlabel('WR time (s)')
	ax.set_ylabel('# of events')
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.minorticks_on()
	plt.show()


	return

def fit_sin(acdc):

	def sin_const_back(x, A, omega, phi, B):
		return A*np.sin(omega*x-phi) + B

	xsin, ysin = acdc.waveforms_sin[0], acdc.waveforms_sin[1]

	sin_lbound, sin_rbound = int(4*(256/25)), int(21*(256/25))
	cut = np.linspace(sin_lbound, sin_rbound, sin_rbound-sin_lbound+1, dtype=int)
	xsin, ysin = xsin[:,cut], ysin[:,cut]

	B0 = np.average(ysin, axis=1)
	A0 = np.max(ysin, axis=1) - B0
	omega0 = np.full_like(B0, 2*np.pi*0.25)
	phi0 = np.zeros_like(B0)
	p0array = np.array([A0, omega0, phi0, B0]).T
	param_bounds = ([0.02, 1.4, -3*np.pi, 0.6], [0.15, 1.75, 3*np.pi, 0.9])

	params = Parameters()
	params.add('A', value=A0.mean(), min=0)
	params.add('omega', value=2*np.pi*0.25)
	params.add('phi', value=0)
	params.add('B', value=B0.mean())

	def residual(params, x, y):
		model = sin_const_back(x, params['A'].value, params['omega'].value, params['phi'].value, params['B'].value)
		return model - y

	popt_vec = np.empty((0, 4), dtype=np.float64)
	skipped = 0
	for xs, ys, p0, tl in zip(xsin, ysin, p0array, acdc.startcap):
		# continue
		try:
			wraparound_ind = 255-tl
			if wraparound_ind < sin_rbound:		# must fit sin wave on leftside of wraparound
				xs, ys = xs[:wraparound_ind-sin_lbound], ys[:wraparound_ind-sin_lbound]
			if len(xsin) < 40:	# if we aren't fitting a full cycle throw event
				raise

			popt, pcov = curve_fit(sin_const_back, xs, ys, p0=p0, bounds=param_bounds)
			# popt, pcov = curve_fit(sin_const_back, xs, ys, p0=p0)

			# result = minimize(residual, params, args=(xs, ys))
			# popt = np.array([result.params['A'].value, result.params['omega'].value, result.params['phi'].value, result.params['B'].value])
			popt_vec = np.vstack((popt_vec, popt))
			# print(popt[0])

			# if tl == 16 and popt[1]/(2*np.pi)*1e3 > 249.:
			# 	fig, ax = plt.subplots()
			# 	ax.scatter(xs, ys, marker='.', color='black', label='Raw data')
			# 	fig_domain = np.linspace(xs[0], xs[-1], 200)
			# 	# ax.plot(fig_domain, sin_const_back(fig_domain, *p0), label='p0')
			# 	ax.plot(fig_domain, sin_const_back(fig_domain, *popt), label='popt', color='red')
			# 	ax.set_xlabel('Sample time (ns)', fontdict=dict(size=14))
			# 	ax.set_ylabel('Voltage (V)', fontdict=dict(size=14))
			# 	ax.legend()
			# 	ax.xaxis.set_ticks_position('both')
			# 	ax.yaxis.set_ticks_position('both')
			# 	plt.minorticks_on()
			# 	plt.show()


		except:
			skipped += 1

	phis = popt_vec[:,2]%(2*np.pi)
	phis = phis/(2*np.pi*0.25)
	phis[phis >= 2.] -= 4.
	omegas = 1e3*popt_vec[:,1]/(2*np.pi)
	fig, ax = plt.subplots()
	_, _, _, image_mesh = ax.hist2d(omegas, phis, bins=200, label=f'ACDC {acdc.acdc_id}')
	fig.colorbar(image_mesh, ax=ax)
	ax.set_xlabel('WR sine frequency (MHz)', fontdict=dict(size=14))
	ax.set_ylabel('250 MHz sine phase (ns)', fontdict=dict(size=14))
	# ax.legend()
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.minorticks_on()
	plt.show()
		
	print(skipped/xsin.shape[0]*100)

	fig, ax = plt.subplots()
	ax.hist(popt_vec[:,0], bins=200)
	# ax.hist(A0, bins=200)
	plt.show()

	fig, ax = plt.subplots()
	ax.hist(popt_vec[:,1]/(2*np.pi)*1e3, bins=200)
	# ax.hist(A0, bins=200)
	plt.show()

	fig, ax = plt.subplots()
	ax.hist(popt_vec[:,2], bins=200)
	# ax.hist(A0, bins=200)
	plt.show()

	fig, ax = plt.subplots()
	ax.hist(popt_vec[:,3], bins=200)
	# ax.hist(A0, bins=200)
	plt.show()


	return



if __name__ == '__main__':
	
	# acdc52 = Acdc('acdc52')
	# acdc52.load_npz('acdc52_stat1_nowrap2_notcal_notnorm.npz')
	# acdc62 = Acdc('acdc62')
	# acdc62.load_npz('acdc62_stat2_nowrap_notimecal_tnormed_bettersin2')
	acdc60 = Acdc('acdc60')
	acdc60.load_npz('acdc60_nowrap2_timecal_notnorm_selftrig3000_1.npz')
	# acdc50 = Acdc('acdc50')
	# acdc50.load_npz('acdc50_nowrap2_notcal_notnorm_selftrigsin_1.npz')

	# fig, ax = plt.subplots()
	# ax.hist(acdc60.p, bins=200)
	# plt.show()

	# fig, ax = plt.subplots()
	# ax.hist(acdc62.chi2[:,2], bins=np.linspace(0,.05,201))
	# plt.show()

	# plot_sinstuff(acdc52)
	# fit_sin(acdc52)

	# plot_sin_time_hists(acdc60)
	plot_omega_phi_sin(acdc60)
	# plot_omega_phi_sin(acdc52_wrap)
	# plot_omega_sin(acdc52)
	# plot_phi_sin(acdc60)
	exit()
	
	acdcs_list = [acdc52, acdc62]
	event_nums = nfold_coincidence(acdcs_list)
	
	acdc_pair = [acdc52, acdc62]
	# twofold_mass_squared(acdc_pair, event_nums, 120)
	self_calibrate(acdc_pair, event_nums)



	


