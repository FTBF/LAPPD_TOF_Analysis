import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline, CubicSpline, PPoly
from scipy.signal import find_peaks, find_peaks_cwt, correlate, correlation_lags
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

@numba.jit(nopython=True)
def gauss_const_back(x, A, c, mu, B):
	return A*np.exp(-c*(x-mu)**2) + B

@numba.jit(nopython=True)
def sin_const_back(x, A, omega, phi, B):
	return A*np.sin(omega*x-phi)+B

@numba.jit(nopython=True)
def sin_const_back_250(x, phi, A, B):
	return A*np.sin(2*np.pi*0.25*x+phi)+B

@numba.jit(nopython=True)
def linear_gauss(x, x0, A, B):
	SIGMA = 1 # in ns
	SLOPE = 0.2 # in 1/ns

	return A* (np.exp(-((x-x0)**2)/(2*SIGMA**2))+ SLOPE*(x-x0)) + B 

#very simple, post-pulse baseline calculator
def find_baseline_simple(ydata, samples_before_end):
	return np.median(ydata[-samples_before_end:])

def find_baseline_std_simple(ydata, samples_before_end):
	return np.std(ydata[-samples_before_end:])

def find_baseline(ydata, samples_from_beginning, sam):

	# 1. Find simple baseline first, as we want to idenfity peaks.
	# 2. After identifying the first peak, take descent from it, forward in time, until the derivative is small enough.(Low pass filtering required in this step.)
	# 3. Take as many samples, forward in time, while the standard deviation decreases. Stop once stdev increases.

	return np.median(ydata[:samples_from_beginning])

def find_baseline_std(ydata, samples_from_beginning):
	return np.std(ydata[:samples_from_beginning])

def roll_timebase(timebase_ns, x_start_cap):
	return np.cumsum(np.roll(timebase_ns, 1-x_start_cap))

def find_peak_time_basic(ydata, y_robust_min, timebase_ns):
	"""Finds the time of the peak of the waveform.
	Arguments:	
		(ndarray)	ydata:		1 dimensional array representing the waveform, after rollover. Peaks must be positive.
		(float)		y_robust_min:peaks are found by comparing the waveform to this value
		(ndarray)	timebase_ns:	the time distance (in nanoseconds) between i th and i+1 th sample in ydata. must have the same length as ydata, and has been rolled, i.e. timebase_ns[0] corresponds to ydata[x_start_cap].
	"""
	#First, we find integer indices of the peaks. This is computationally cheap. After that, we will interpolate to find the peak time with 1 ps resolution.
	peaks, props = find_peaks(ydata, height=0.8*(-y_robust_min), width = 10, distance=20)#These numbers heavily depend on LAPPD characteristics and are subject to change.
	#peaks_cwt = find_peaks_cwt(vector = ydata_rolled, width = 10)#Alternative method.
	
	return [timebase_ns[int(peak)] for peak in peaks], peaks, props["peak_heights"]
	
	#Do not use this function. The output format is different from the other peak finding functions.
def find_peak_time(ydata, y_robust_min, timebase_ns):
	"""Finds the time of the peak of the waveform.
	Arguments:	
		(ndarray)	ydata:		1 dimensional array representing the waveform, after rollover. Peaks must be positive.
		(float)		y_robust_min:peaks are found by comparing the waveform to this value
		(ndarray)	timebase_ns:	the time distance (in nanoseconds) between i th and i+1 th sample in ydata. must have the same length as ydata, and has been rolled, i.e. timebase_ns[0] corresponds to ydata[x_start_cap].
	"""
	peaks, peaks_index = find_peak_time_basic(ydata, y_robust_min, timebase_ns)
	#Now we curve_fit to find the peak time with 1 ps resolution.
	if(y_robust_min >0):
		print("Warning: y_robust_min is positive. This is not expected for LAPPD waveforms.")
	for i, peak in enumerate(peaks):
		p0 = [peak, y_robust_min, 0]
		param_scale = [25, np.abs(y_robust_min), 0.1*np.abs(y_robust_min)]
		start_cap = np.max([0, peaks_index[i]-20])
		end_cap = np.min([len(ydata), peaks_index[i]+20])
		popt, pcov = curve_fit(linear_gauss, xdata = timebase_ns[start_cap:end_cap], ydata = ydata[start_cap:end_cap], p0=p0, bounds=([peak-1, y_robust_min*1.1, y_robust_min*(0.1)], [peak+1, y_robust_min*0.6, -y_robust_min*0.1]), x_scale = param_scale)
		peaks[i] = popt[0]
	return peaks

def find_peak_time_inflection(ydata, y_robust_min, timebase_ns, forward_samples = 30, threshold = 0.3, sample_distance = 0.01, trailing_edge_limit = 180):
	"""Finds the time of the peak of the waveform.
	Arguments:	
		(ndarray)	ydata:		1 dimensional array representing the waveform, after rollover. Peaks must be positive.
		(float)		y_robust_min:peaks are found by comparing the waveform to this value
		(ndarray)	timebase_ns:	the time distance (in nanoseconds) between i th and i+1 th sample in ydata. must have the same length as ydata, and has been rolled, i.e. timebase_ns[0] corresponds to ydata[x_start_cap].
		(int)		forward_samples:	number of samples to take before the peak to find the inflection point.
		(float)		threshold:		maximum slope times the threshold is the slope of the waveform at the returned peak time.
	"""
	peaks, peaks_index = find_peak_time_basic(ydata, y_robust_min, timebase_ns)

	
	if(y_robust_min >0):
		print("Warning: y_robust_min is positive. This is not expected for LAPPD waveforms.")
	if(peaks.__len__() != 2):
		raise ValueError("This function is only implemented for two peaks.")
	#First peak corresponds to the leading edge. Second peak corresponds to the trailing edge.
	# We find the inflection point of the leading edge, which is the impact point. Then, the impact point of the trailing edge is computed by autocorrelation.
	impact_points = np.full(shape = 2,fill_value= -1.0)
	################################################## Leading edge ##################################################
	#Take a fixed number of samples before the peak and spline represent them.
	start_cap = np.max([0, peaks_index[0]-forward_samples])
	end_cap = peaks_index[0]

	#https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html#tutorial-interpolate-splxxx
	#https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.roots.html#scipy.interpolate.UnivariateSpline.roots
	#For the smoothing parameter s, we are assuming that the standard deviation of ydata is 1 mV. This is a rough estimate.
	spline_tuple = splrep(timebase_ns[start_cap:end_cap], ydata[start_cap:end_cap], k=3, s=forward_samples)
	
	bsplinePoly = PPoly.from_spline(spline_tuple)
	bsplinePolyPrime = bsplinePoly.derivative()
	#The derivative of the waveform is used to find the inflection point.

	#Measure the greatest slope of the waveform. First take the derivative of the spline. Then find the maximum of the derivative.
	#The maximum of the derivative is the inflection point.
	inflection_points = bsplinePolyPrime.derivative().roots()

	#Now we find the inflection point with the greatest slope.
	max_slope_point = inflection_points[np.argmax(bsplinePolyPrime(inflection_points))]
	max_slope = bsplinePolyPrime(max_slope_point)

	#Min slope point is one of the inflection points or the beginning of the waveform. We make sure that we don't identify the peak as the minimum slope point.
	#There is a tricky math here because we want to find points where the absolute value of the slope is minimized.
	min_slope_point = inflection_points[np.argmin(np.abs(bsplinePolyPrime(inflection_points)))]
	min_slope = bsplinePolyPrime(min_slope_point)
	if(abs(min_slope)>abs(bsplinePolyPrime(timebase_ns[start_cap]))):
		min_slope_point = timebase_ns[start_cap]
		min_slope = bsplinePolyPrime(timebase_ns[start_cap])


	#Make sure that slope is sufficiently small at the minimum slope point.
	if(abs(max_slope)*threshold < abs(min_slope)):
		raise ValueError("The slope of the waveform in the window is too high. Try increasing the forward_samples parameter.")
	else:
		#Solve the spline at the threshold slope to find the impact time.
		impact_candidates = bsplinePolyPrime.solve(max_slope*threshold, extrapolate=False)
		#Find the impact time between the minimum slope point and the maximum slope point.
		impact_times_filtered = impact_candidates[(impact_candidates>min_slope_point) & (impact_candidates<max_slope_point)]
		impact_points[0] = np.max(impact_times_filtered)
		
	################################################## End of Leading edge ##################################################
	################################################## Trailing edge ##################################################

	impact_points[1] = trailing_edge_autocorrelation(impact_points[0], peaks, ydata, sample_distance, timebase_ns, forward_samples, trailing_edge_limit)
	################################################## End of Trailing edge ##################################################


	return impact_points, peaks


def trailing_edge_autocorrelation(first_impact_point, peaks, ydata, sample_distance, timebase_ns, forward_samples, trailing_edge_limit):
	################################################## Trailing edge ##################################################

	#Create autocorrelation function with sampling step = 10 ps and sampling window = [first impact point, first impact point + 6 ns]
	#The sliding window is [first max slope point, first peak time], sampled at 10 ps from bsplinePoly.
	sliding_window_samples = np.arange(first_impact_point, peaks[0], sample_distance)

	#Take a wide window containing the leading edge and the trailing edge and spline represent them.

	start_cap = 0
	end_cap = trailing_edge_limit
	#For the smoothing parameter s, we are assuming that the standard deviation of ydata is 1 mV. This is a rough estimate.
	spline_tuple = splrep(timebase_ns[start_cap:end_cap], ydata[start_cap:end_cap], k=3, s=forward_samples)
	
	bsplinePoly2 = PPoly.from_spline(spline_tuple)

	#Sliding window
	tmp  = np.apply_along_axis(bsplinePoly2, 0, sliding_window_samples)
	#Reference window
	tmp2 = np.apply_along_axis(bsplinePoly2, 0, np.arange(first_impact_point, timebase_ns[end_cap], sample_distance))
	#Be careful with the order of the arguments. The first argument is the reference waveform, and the second argument is the sliding waveform.
	autocorr = correlate(tmp2, tmp)
	autocorr_lags = correlation_lags(np.size(tmp2), np.size(tmp))

	#Some debugging plots.
	# plt.plot(autocorr_lags * sample_distance + impact_points[0], autocorr / np.max(autocorr) * 100)
	# plt.plot(np.arange(impact_points[0], rolled_timebase[end_cap], sample_distance), tmp2)
	# plt.show()

	#Now derive a spline from the autocorrelation samples, and find the peak of the spline.
	autocorr_spline_tuple = splrep(autocorr_lags, autocorr, k=3)
	autocorr_bspline = PPoly.from_spline(autocorr_spline_tuple)
	autocorr_dbspline = autocorr_bspline.derivative()
	autocorr_extrema = autocorr_dbspline.solve(0, extrapolate=False)

	#Candidates are the two extremas with the largest autocorrelation value.
	candidates = autocorr_extrema[np.argsort(autocorr_bspline(autocorr_extrema))][-2:]
	candidates.sort()
	#Measure the distance between the leading extremum and the first impact point. From the distance, we can find the second impact point from the second extremum.
	return first_impact_point + (candidates[1] - candidates[0])*sample_distance
	################################################## End of Trailing edge ##################################################


#Do not use this function. It is not implemented.
def find_peak_time_10_90(ydata, y_robust_min, timebase_ns, forward_samples = 20):
	"""Finds the time of the peak of the waveform. This function currently does not work: read todo comment below.
	Arguments:	
		(ndarray)	ydata:		1 dimensional array representing the waveform, after rollover. Peaks must be positive.
		(float)		y_robust_min:peaks are found by comparing the waveform to this value
		(ndarray)	timebase_ns:	the time distance (in nanoseconds) between i th and i+1 th sample in ydata. must have the same length as ydata, and has been rolled, i.e. timebase_ns[0] corresponds to ydata[x_start_cap].
		(int)		forward_samples:	number of samples to take before the peak to find the 10% and 90% levels.
	"""
	peaks, peaks_index = find_peak_time_basic(ydata, y_robust_min, timebase_ns)

	if(y_robust_min >0):
		print("Warning: y_robust_min is positive. This is not expected for LAPPD waveforms.")
	for i, peak in enumerate(peaks):
		#Take a fixed number of samples before the peak and spline represent them.
		start_cap = np.max([0, peaks_index[i]-forward_samples])
		end_cap = peaks_index[i]

		#https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html#tutorial-interpolate-splxxx
		#https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.roots.html#scipy.interpolate.UnivariateSpline.roots
		spline_tuple = splrep(timebase_ns[start_cap:end_cap], ydata[start_cap:end_cap], k=3, s=forward_samples)
		bsplinePoly = PPoly.from_spline(spline_tuple)

		#Find the 10% and 90% of the peak amplitude.
		#TODO: Note: We implemented 30% levels because the 10% level was too close to the noise floor.
		peak_amplitude = bsplinePoly(timebase_ns[end_cap])
		peak_30 = peak_amplitude*0.3
		peak_90 = peak_amplitude*0.9
		#Find the time at which the waveform crosses the 10% and 90% levels.
		print("peak30 solve:", bsplinePoly.solve(peak_30, extrapolate=False))
		print("peak90 solve:", bsplinePoly.solve(peak_90, extrapolate=False))
		peak_30_time = bsplinePoly.solve(peak_30, extrapolate=False)[0]
		peak_90_time = bsplinePoly.solve(peak_90, extrapolate=False)[0]
		#Draw a line between the 10% and 90% levels and find the intersection with the y=0 line.
		#The intersection is the impact time.
		impact_time = peak_30_time + (peak_90_time - peak_30_time) * peak_30 / (peak_30 - peak_90)
		peaks[i] = impact_time
	return peaks

def find_peak_time_CFD(ydata, y_robust_min, timebase_ns, forward_samples = 25, backward_samples = 0, threshold = 0.22, forward_ns_effective_amplitude = 0, sample_distance = 0.01, trailing_edge_limit = 180):
	"""Finds the time of the peak of the waveform.
	Arguments:	
		(ndarray)	ydata:		1 dimensional array representing the waveform, after rollover. Peaks must be positive.
		(float)		y_robust_min:peaks are found by comparing the waveform to this value
		(ndarray)	timebase_ns:	the time distance (in nanoseconds) between i th and i+1 th sample in ydata. must have the same length as ydata, and has been rolled, i.e. timebase_ns[0] corresponds to ydata[x_start_cap].
		(int)		forward_samples:	number of samples to take before the peak to find the inflection point.
		(float)		threshold:		maximum slope times the threshold is the slope of the waveform at the returned peak time.
	"""
	peaks, peaks_index, peak_height = find_peak_time_basic(ydata, y_robust_min, timebase_ns)

	
	if(y_robust_min >0):
		print("Warning: y_robust_min is positive. This is not expected for LAPPD waveforms.")
	if(peaks.__len__() != 2):
		raise ValueError("This function is only implemented for two peaks.")
	#First peak corresponds to the leading edge. Second peak corresponds to the trailing edge.
	# We find the CFD point of the leading edge, which is the impact point. Then, the impact point of the trailing edge is computed by autocorrelation.
	impact_points = np.full(shape = 2,fill_value= -1.0)
	################################################## Leading edge ##################################################
	#Take a fixed number of samples around the peak and spline represent them.
	start_cap = np.max([0, peaks_index[0]-forward_samples])
	end_cap = peaks_index[0] + backward_samples

	#https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html#tutorial-interpolate-splxxx
	#https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.roots.html#scipy.interpolate.UnivariateSpline.roots
	#For the smoothing parameter s, we are assuming that the standard deviation of ydata is 1 mV. This is a rough estimate.
	spline_tuple = splrep(timebase_ns[start_cap:end_cap], ydata[start_cap:end_cap], k=3, s=forward_samples)
	
	bsplinePoly = PPoly.from_spline(spline_tuple)
	#The derivative of the waveform is used to find the peak amplitude
	peaks_from_derivation = bsplinePoly.derivative().roots()

	primary_peak_from_derivation = peaks_from_derivation[np.argmax(bsplinePoly(peaks_from_derivation))]

	#This amplitude is feed back to Acdc reduced quantities for analysis.
	amplitude = bsplinePoly(primary_peak_from_derivation- forward_ns_effective_amplitude)

	#The start_value serves as a local baseline for the CFD. It is the average of the first 5 samples of the waveform in the beginning of the window.
	start_value = np.mean([bsplinePoly(timebase_ns[start_cap+i]) for i in range(5)])

	#Make sure that value of the waveform is sufficiently small at the beginning of the window.
	if(amplitude*threshold < start_value):
		raise ValueError("The slope of the waveform in the window is too high. Try increasing the forward_samples parameter.")
	else:
		#Solve the spline at the threshold slope to find the impact time.
		impact_candidates = bsplinePoly.solve(start_value+(amplitude - start_value)*threshold, extrapolate=False)
		#Find the impact time between the minimum slope point and the maximum slope point.
		impact_times_filtered = impact_candidates[(impact_candidates>timebase_ns[start_cap]) & (impact_candidates<timebase_ns[peaks_index[0]])]
		impact_points[0] = np.max(impact_times_filtered)#If there are still multiple candidates, take the latest one. Beginning of the waveform could have unwanted ripples.
		
	################################################## End of Leading edge ##################################################
	################################################## Trailing edge ##################################################

	impact_points[1] = trailing_edge_autocorrelation(impact_points[0], peaks, ydata, sample_distance, timebase_ns, forward_samples, trailing_edge_limit)
	################################################## End of Trailing edge ##################################################


	return impact_points, peaks, amplitude

def find_sine_phase(ydata, timebase_ns, ydata_max, samples_after_zero, samples_before_end):
		"""
		Finds the phase of a sine wave in the waveform.
		x must refer to a point in the waveform that is not in the trigger region, AFTER the rollover.

		Arguments:
			(ndarray)	ydata:		1 dimensional array representing the waveform, after rollover.
			(ndarray)	timebase_ns:	the time distance (in nanoseconds) between i th and i+1 th sample in ydata. must have the same length as ydata, and has been rolled, i.e. timebase_ns[0] corresponds to ydata[x_start_cap].
			(float)		x:			the time of the waveform at which the phase is to be found.
			(int)		samples_before_end:	number of samples to exclude before the end, to avoid the trigger region.
		"""
		#Sine wave frequency in gigahertz. Note that sin_const_back_250 must be adjusted if this is changed.
		FREQ = 0.25
		#Fit the sine wave using curve_fit
		p0 = [0, ydata_max, 0]#Amplitude, phase, offset
		param_scale = [np.pi, ydata_max , ydata_max*0.1]#Rough scale of the parameters, used to adjust the step size in the minimization routine.
		single_cycle = int(1/FREQ / (25/256))
		start_point = samples_after_zero + single_cycle
		#The bounds are empirically set to be reasonable for LAPPD waveforms. In particular, the amplitude has to be positive. 
		#The first fit is done over a single cycle of the waveform, to get a rough estimate of the parameters.
		popt, pcov = curve_fit(sin_const_back_250, xdata = timebase_ns[start_point:start_point+single_cycle], ydata = ydata[start_point:start_point+single_cycle], p0=p0, bounds=([-2*np.pi, ydata_max*0.5, -ydata_max*0.5], [2*np.pi, ydata_max*1.5, ydata_max*0.5]), x_scale = param_scale)

		#The second fit is done with the amplitude and offset fixed to the value found in the first fit. This is done to improve the fit.
		phi, pcov = curve_fit(lambda x, phi: sin_const_back_250(x, phi, popt[1], popt[2]), xdata = timebase_ns[samples_after_zero:-samples_before_end], ydata = ydata[samples_after_zero:-samples_before_end], p0=popt[0], bounds=(-2*np.pi, 2*np.pi))
		return [phi, popt[1], popt[2]]

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