import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline, CubicSpline, PPoly
from scipy.signal import find_peaks, find_peaks_cwt
from matplotlib import pyplot as plt
from scipy.integrate import trapezoid
import numba

# Some helper functions
def convert_to_list(some_object):
	"""Converts an object into a single-element list if the object is not already a list. Helps when a function works by going through elements of a list, but you want to pass a single element to that function.
	Arguments:
		(any) some_object: what you want to convert to a list (if not already a list)
	"""

	if not isinstance(some_object, list) and not isinstance(some_object, np.ndarray):
		some_object = [some_object]

	return some_object

def find_extrema_spline(xdata, ydata):

	spline_tuple = splrep(xdata, ydata, k=3, s=10000)
	bspline = BSpline(*spline_tuple)
	dbspline = bspline.derivative()
	dcubic_spline = CubicSpline(xdata, dbspline(xdata))
	extrema = dcubic_spline.solve(0, extrapolate=False)

	return extrema, bspline

def compute_sliding_function(xdata, ydata, lbound, rbound, stat_spline, func, slide_increment=1, FAST=False):
	"""Handles the sliding/computing part of finding the autocorrelation function or lag-based chi-squared.
		(ndarray)	xdata:				xdata of the waveform, i.e. the sample times
		(ndarray)	ydata				ydata of the waveform, i.e. the voltages
		(float)		lbound				left bound in the subdomain of xdata you wish to compute the sliding function over
		(float)		rbound				right bound in the subdomain of xdata you wish to compute the sliding function over
		(BSpline)	stat_spline			BSpline object interpolating the subrange of ydata over the subdomain of xdata
		(float)		func				the sliding function you wish to apply. Takes as inputs stationary ydata and slid
										ydata. For autocorrelation returns the integral of stat_ydata*slid_ydata. For chi-squared
										returns integral of (stat_ydata-slid_ydata)**2
		(float)		slide_increment=1	effectively the number of indices you're sliding the data by. Default is 1 (~250 ps), but 
										can be set to less than 1.
	"""

	if FAST:
		abs_ydata = np.absolute(ydata)
		ydata_max = np.amax(abs_ydata)
		indices, _ = find_peaks(abs_ydata, height=0.6*ydata_max, distance=5)
		limit = indices[-1] - indices[0]
		if limit == 0:
			limit = 256
		else:
			limit += 15
	else:
		limit = 256

	func_vals = []
	lags = []
	lag = 0
	lag_factor = 25/256
	while lag < limit:

		xdata_sliding = np.copy(xdata) - lag_factor*lag*np.ones_like(xdata)
		indices_inbounds = np.linspace(0,255,256,dtype=int)[(xdata_sliding >= lbound) & (xdata_sliding <= rbound)]
		if len(indices_inbounds) == 0:
			lag += slide_increment
			continue

		xdata_shifted_inbounds = xdata_sliding[indices_inbounds]
		ydata_inbounds = ydata[indices_inbounds]
		
		if not indices_inbounds[0] == 0:
			r_ind = indices_inbounds[0]
			l_ind = r_ind - 1
			lbound_yval = np.interp(lbound, [xdata_sliding[l_ind], xdata_sliding[r_ind]], [ydata[l_ind], ydata[r_ind]])
			xdata_shifted_inbounds = np.insert(xdata_shifted_inbounds, 0, lbound)
			ydata_inbounds = np.insert(ydata_inbounds, 0, lbound_yval)

		if not indices_inbounds[-1] == 255:
			l_ind = indices_inbounds[-1]
			r_ind = l_ind + 1
			rbound_yval = np.interp(rbound, [xdata_sliding[l_ind], xdata_sliding[r_ind]], [ydata[l_ind], ydata[r_ind]])
			xdata_shifted_inbounds = np.append(xdata_shifted_inbounds, rbound)
			ydata_inbounds = np.append(ydata_inbounds, rbound_yval)
							
		ydata_stationary = stat_spline(xdata_shifted_inbounds)

		func_vals.append(func(ydata_inbounds, ydata_stationary, xdata_shifted_inbounds))
		lags.append(lag_factor*lag)

		# if round(lag_factor*lag, 3) == 1.66:
		# 	fig, ax = plt.subplots()
		# 	ax.scatter(xdata, ydata, marker='.', label='Stationary')
		# 	ax.plot(domain_for_plot, stat_spline(domain_for_plot), label='Stationary spline')
		# 	ax.scatter(xdata_shifted_inbounds, ydata_inbounds, marker='.', label='Lagged')
		# 	ax.scatter(xdata_shifted_inbounds, ydata_stationary, marker='.', color='red')
		# 	for i, x_val in enumerate(xdata_shifted_inbounds):
		# 		if ydata_inbounds[i] > ydata_stationary[i]:
		# 			lower_val = ydata_stationary[i]
		# 			higher_val = ydata_inbounds[i]
		# 		else:
		# 			lower_val = ydata_inbounds[i]
		# 			higher_val = ydata_stationary[i]
		# 		ax.axvline(x_val, ymin=(lower_val+1200)/1100, ymax=(higher_val+1200)/1100, color='gray')
		# 	ax.axvline(lbound, color='green')
		# 	ax.axvline(rbound, color='green')
		# 	# ax.axvline(xdata_shifted_inbounds[3], color='C1')
		# 	ax.text(2.663, -1180, f'Lag: {round(lag_factor*lag, 3)} ns', fontdict=dict(size=12))
		# 	ax.set_xlim(lbound-0.2, rbound+0.2)
		# 	# ax.set_xlim(1.5, 9)
		# 	ax.set_ylim(-1200, -100)
		# 	ax.legend(loc='lower right', framealpha=1)
		# 	ax.set_xlabel('Sample time (ns)')
		# 	ax.set_ylabel('ADC count')
		# 	ax.xaxis.set_ticks_position('both')
		# 	ax.yaxis.set_ticks_position('both')
		# 	plt.minorticks_on()

		# 	fig2, ax2 = plt.subplots()
		# 	ax2.scatter(xdata_shifted_inbounds, (ydata_inbounds-ydata_stationary)**2, color='black', marker='.')
		# 	for i in range(len(xdata_shifted_inbounds)-1):
		# 		x_pols = [xdata_shifted_inbounds[i], xdata_shifted_inbounds[i], xdata_shifted_inbounds[i+1], xdata_shifted_inbounds[i+1]]
		# 		y_pols = [0, ((ydata_inbounds-ydata_stationary)**2)[i], ((ydata_inbounds-ydata_stationary)**2)[i+1], 0]
		# 		ax2.fill(x_pols, y_pols, color='C0', alpha=0.45, edgecolor='black')
		# 	ax2.text(2.813,5.35e5, f'Lag: {round(lag_factor*lag, 3)} ns', fontdict=dict(size=12))
		# 	ax2.text(2.813, 4.9e5, 'Integral value: {:.2e}'.format(func_vals[-1]), fontdict=dict(size=12))
		# 	ax2.set_xlabel('Sample time (ns)')
		# 	ax2.set_ylabel('Least squares values')
		# 	ax2.xaxis.set_ticks_position('both')
		# 	ax2.yaxis.set_ticks_position('both')
		# 	plt.minorticks_on()

		# 	plt.show()

		lag += slide_increment

	func_vals = np.array(func_vals)
	lags = np.array(lags)

	return lags, func_vals

def find_leading_edge(xdata, ydata, SPLINE_CFD=False):

	# Determines the indices of the peaks in the prompt and reflected pulses
	height_cutoff = -0.6*ydata.max()
	distance_between_peaks = 20		# in units of indices
	peak_region_radius = 15			# in units of indices
	# removed -1*ydata for vcc calibrated stuff
	peaks_rough = find_peaks(ydata, height=height_cutoff, distance=distance_between_peaks)[0]
	prompt_peak_index, reflect_peak_index = np.sort(peaks_rough[ydata[peaks_rough].argsort()[0:2]])

	# Creates subregions of data around the reflect peak
	reflect_lbound = reflect_peak_index - int((reflect_peak_index-prompt_peak_index)/2)-5 # lower bound is a bit left of the midway between peaks
	reflect_ubound = reflect_peak_index + 6
	ydata_subrange = ydata[reflect_lbound:reflect_ubound]
	reflect_subdomain = xdata[reflect_lbound:reflect_ubound]
	peak_region_lower, peak_region_upper = xdata[reflect_peak_index-4], xdata[reflect_peak_index+4]

	# Solves for the extrema of the reflect peak
	# spline_tuple = splrep(reflect_subdomain, ydata_subrange, k=3, s=10000)
	spline_tuple = splrep(reflect_subdomain, ydata_subrange, k=3)
	reflect_bspline = BSpline(*spline_tuple)
	reflect_dbspline = reflect_bspline.derivative()
	reflect_dcubic_spline = CubicSpline(reflect_subdomain, reflect_dbspline(reflect_subdomain))
	extrema = reflect_dcubic_spline.solve(0, extrapolate=False)
	fig, ax = plt.subplots()
	ax.plot(reflect_subdomain, reflect_bspline(reflect_subdomain))
	for thing in extrema:
		ax.axvline(thing)
	ax.axvline(peak_region_lower, label='lower', color='red')
	ax.axvline(peak_region_upper, label='upper', color='purple')
	ax.legend()
	plt.show()
	print(extrema)
	reflect_peak_max = reflect_bspline(extrema[(extrema > peak_region_lower) & (extrema < peak_region_upper)])	# finds the extrema that is near our original find_peaks value
	reflect_peak_max = reflect_peak_max[0]
	reflect_peak_min_val = reflect_bspline(extrema[0]) + 0.1*(reflect_peak_max - reflect_bspline(extrema[0]))

	# repeating the spline for the prompt peak now
	prompt_lbound = prompt_peak_index - 20
	if prompt_lbound < 0:
		prompt_lbound = 0
	prompt_ubound = prompt_peak_index + 4
	prompt_subrange = ydata[prompt_lbound:prompt_ubound]
	prompt_subdomain = xdata[prompt_lbound:prompt_ubound]
	# prompt_tuple = splrep(prompt_subdomain, prompt_subrange, k=3, s=10000)
	prompt_tuple = splrep(prompt_subdomain, prompt_subrange, k=3)
	prompt_bspline = BSpline(*prompt_tuple)
	prompt_cubic_spline = CubicSpline(prompt_subdomain, prompt_bspline(prompt_subdomain))
	prompt_dbspline = prompt_bspline.derivative()
	prompt_dcubic_spline = CubicSpline(prompt_subdomain, prompt_dbspline(prompt_subdomain))
	prompt_extrema = prompt_dcubic_spline.solve(0)
	peak_region_lower, peak_region_upper = xdata[prompt_peak_index-3], xdata[prompt_peak_index+3]
	prompt_peak_max = prompt_bspline(prompt_extrema[(prompt_extrema > peak_region_lower) & (prompt_extrema < peak_region_upper)])
	prompt_peak_max = prompt_peak_max[0]

	fig, ax = plt.subplots()
	ax.scatter(xdata, ydata)
	ax.plot(prompt_subdomain, prompt_bspline(prompt_subdomain))
	ax.axhline(reflect_peak_min_val, color='red', label='min')
	ax.axhline(0.9*prompt_peak_max, color='purple', label='max')
	ax.legend()
	plt.show()

	# Computes the integral bounds
	lbound = prompt_cubic_spline.solve(reflect_peak_min_val, extrapolate=False)[0]
	rbound = prompt_cubic_spline.solve(0.9*prompt_peak_max, extrapolate=False)[0]

	if SPLINE_CFD:
		reflect_cubicspline = CubicSpline(reflect_subdomain, reflect_bspline(reflect_subdomain))
		r_intersects = reflect_cubicspline.solve(reflect_peak_min_val, extrapolate=False)
		reflect_cfd_pos = (r_intersects[r_intersects < xdata[reflect_peak_index]])[-1]
		return lbound, reflect_cfd_pos
	else:
		return lbound, rbound, prompt_cubic_spline

@numba.jit(nopython=True)
def gauss_const_back(x, A, c, mu, B):
	return A*np.exp(-c*(x-mu)**2) + B

@numba.jit(nopython=True)
def sin_const_back(x, A, omega, phi, B):
	return A*np.sin(omega*x-phi)+B

@numba.jit(nopython=True)
def sin_const_back_250(x, A, phi, B):
	return A*np.sin(2*np.pi*0.25*x-phi)+B

#very simple, post-pulse baseline calculator
def find_baseline_simple(ydata, samples_before_end):
	return np.median(ydata[-samples_before_end:])

def find_baseline_std_simple(ydata, samples_before_end):
	return np.std(ydata[-samples_before_end:])
	
def find_peak_time(ydata, y_robust_min, x_start_cap, timebase_ns):
	"""Finds the time of the peak of the waveform.
	Arguments:	
		(ndarray)	ydata:		1 dimensional array representing the waveform, after rollover.
		(float)		y_robust_min:peaks are found by comparing the waveform to this value
		(int)		x_start_cap:index of ydata at which the waveform starts, i.e. most temporally advanced sample in the waveform
		(ndarray)	timebase_ns:	the time distance (in nanoseconds) between i th and i+1 th sample in ydata. must have the same length as ydata, and has not been rolled, i.e. timebase_ns[0] corresponds to ydata[x_start_cap].
	"""
	timebase_sum_rolled = np.cumsum(np.roll(timebase_ns, -x_start_cap))
	#First, we find integer indices of the peaks. This is computationally cheap. After that, we will interpolate to find the peak time with 1 ps resolution.
	peaks, props = find_peaks(ydata, height=0.8*(-y_robust_min), width = 10, distance=20)#These numbers heavily depend on LAPPD characteristics and are subject to change.
	#peaks_cwt = find_peaks_cwt(vector = ydata_rolled, width = 10)#Alternative method.
	
	return [timebase_sum_rolled[int(peak)] for peak in peaks]

def find_sine_phase(ydata, timebase_ns, x_start_cap, x):
		"""
		Finds the phase of a sine wave in the waveform.
		x must refer to a point in the waveform that is not in the trigger region, AFTER the rollover.
		"""
		#Sine wave frequency in gigahertz. Note that sin_const_back_250 must be adjusted if this is changed.
		FREQ = 0.25


		ydata_rolled = np.roll(ydata, -x_start_cap)
		timebase_rolled = np.roll(timebase_ns, -x_start_cap)
		#Fit the sine wave using curve_fit
		p0 = [0.1, 0, 0.1]
		popt, pcov = curve_fit(sin_const_back_250, timebase_rolled, ydata_rolled, p0=p0)
	
		return np.remainder(popt[2] + 2*np.pi*FREQ*x, 2*np.pi)

def calc_vpos(xv, yv, mu0):
		p0 = [-0.25*yv.max(), 0.01, mu0, 0.8]
		popt, pcov = curve_fit(gauss_const_back, xv, yv, p0=p0)
		# fig, ax = plt.subplots()
		# ax.scatter(xv, yv, marker='.', color='black')
		# domain = np.linspace(xv[0], xv[-1], 200)
		# ax.plot(domain, gauss_const_back(domain, *p0), color='green')
		# ax.plot(domain, gauss_const_back(domain, *popt), color='red')
		# plt.show()

		return popt[2]

def leading_edge_bounds(xh, yh):

		yh_temp = -yh + yh.max()
		min_height = 0.6*yh_temp.max()
		peak_dist = 20
		
		peaks_rough = find_peaks(yh_temp, height=min_height, distance=peak_dist)[0]
		prompt_ind, reflect_ind = peaks_rough[peaks_rough > 8][0:2]

		lbound = prompt_ind - 25
		if lbound < 0:
			lbound = 0
		rbound = prompt_ind + 4
		subdomain = xh[lbound:rbound]
		subrange = yh[lbound:rbound]
		cspline = CubicSpline(subdomain, subrange, extrapolate=False, bc_type='natural')

		ymin, ymax = yh[lbound], yh[prompt_ind]
		lbound_y = ymin - 0.1*(ymin-ymax)
		rbound_y = ymin - 0.9*(ymin-ymax)
		
		lbound = cspline.solve(lbound_y, extrapolate=False)[0]
		rbound = cspline.solve(rbound_y, extrapolate=False)[0]
		return lbound, rbound, reflect_ind

def find_lsquares(xh, yh, lbound, rbound, offsets):

	bspline_tup = splrep(xh, yh, k=3)
	bspline = BSpline(*bspline_tup)

	x = np.linspace(lbound, rbound, 10)
	y = bspline(x)

	x_shift = np.vstack([x + dt for dt in offsets])
	y_shift = bspline(x_shift)

	least_squares = (y_shift - y)**2
	avg_lsquares = trapezoid(least_squares, x, axis=1)

	return avg_lsquares