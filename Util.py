import math
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
import bitstruct as bitstruct
import scipy
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import os.path
import sys
#Util Class is used for generating board calibration files and various measurements that are not directly used in TOF analysis.
#1. Voltage curve calibration file.
#2. Timebase calibration file.
#3. Phase distribution between channels.
class Util:
	def __init__(self, board_config=None):
		self.measurement_config = Util.load_config(board_config)
		self.df = pd.DataFrame(columns=["ch", "times", "voltage_count_curves"])
		if(os.path.isfile(self.measurement_config["calibration_file"])):
			self.df = pd.read_hdf(self.measurement_config["calibration_file"])
			print("Calibration file loaded.")
		
		self.trigger_pos = []
		
	def sine(x, A, B, omega, phi):
		return A * np.sin(omega*x + phi) + B

	#Receives a radian and returns the value wrapped to [-pi, pi]
	def wrap(x):
		return np.remainder(x+math.pi, 2*math.pi)-math.pi
	def getDataRaw(fname):
		data = []
		format=bitstruct.compile("p4u12u12u12u12u12"*(256*int(30/5)))
		swapformat="8"*(256*int(30/5))
		with open(fname, "rb") as f:
			line = f.read((1+4+int(256*30/5))*8)
			while len(line) == (1+4+int(256*30/5))*8:
				data.extend(format.unpack(bitstruct.byteswap(swapformat, line[5*8:])))
				line = f.read((1+4+int(256*30/5))*8)
		data = np.array(data)
		return data.reshape([-1, 30, 256])

	def getData(fnames):
		data = []
		for fname in fnames:
			with open(fname) as f:
				for line in f:
					data.extend(line.split()[1:31])
		data = np.array(data, dtype='float')
		return data.reshape([-1, 256, 30])    
	
	def load_config(fn):
		if(fn is None):
			print("No calibration measurement config provided, loading default configuration")
			return Util.load_config("configs/calib_measurement_config.yml")

		with open(fn, "r") as stream:
			try:
				return yaml.safe_load(stream)
			except Exception as exc:
				print("Had an exception while reading yaml file for util config: %s"%exc)

	def save(self):
		self.df.to_hdf(self.measurement_config["calibration_file"], "calibration_jin", "w")
		#pickle.dump(voltage_curve, open(self.measurement_config["voltage_curve"]["output"], "wb"))
		#pd.DataFrame(self.voltage_curve).to_hdf(self.measurement_config["voltage_curve"]["output"]) Cannot output 3d array to hdf5 file?!!
	
	#Reads a series of raw data files and saves a voltage curve calibration file.
	def create_voltage_curve(self):
		voltage_curve = []
		x_vals = [i for i in range(self.measurement_config["voltage_curve"]["start"], self.measurement_config["voltage_curve"]["end"], self.measurement_config["voltage_curve"]["step"])]
		for i in x_vals:
			pedData = Util.getDataRaw(self.measurement_config["voltage_curve"]["prefix"]+str(i)+self.measurement_config["voltage_curve"]["suffix"])
			voltage_curve.append([np.full((30,256), i), pedData.mean(0)])
			#voltage_curve.append([np.full((30,256), i), np.full((30,256), i)])
			print(i)
		#voltage_curve is a list of size (# of measurement points), each measurement point is a 2x30(ch)x256(# of capacitors) array of voltage values.
		voltage_curve = np.array(voltage_curve).transpose(2,3,0,1)
		#self.df is a dataframe with 30 rows of voltage_count_curves, which is an array of 256(# of capacitors)*(# of measurement points)*[voltage, ADC count], # of measurement points typically being 256.
		for i in range(30):
			self.df.loc[i, "voltage_count_curves"] = voltage_curve[i]
		self.save()
	def find_trigger_pos(self, sineData):
		nevents = self.measurement_config["timebase"]["nevents"]
		trigger_pos = [0]*nevents
		for e in range(0,nevents):
			pulse = sineData[e,:,:]
			max = 0
			for iCap in range(256):
				#Find the most indifferentiable point of the pulse, window of 5 samples.
					cap0 = (iCap+254)%256
					cap1 = iCap
					cap2 = (iCap+2)%256
					diff01 = np.abs(pulse[11, cap1] - pulse[11, cap0]-pulse[11, cap1]+pulse[11, cap0])
					diff02 = np.abs(pulse[11, cap1] - pulse[11, cap0]-pulse[17, cap1]+pulse[17, cap0])
					diff03 = np.abs(pulse[11, cap1] - pulse[11, cap0]-pulse[23, cap1]+pulse[23, cap0])
					diff04 = np.abs(pulse[11, cap1] - pulse[11, cap0]-pulse[29, cap1]+pulse[29, cap0])
					diff11 = np.abs(pulse[11, cap2] - pulse[11, cap1]-pulse[11, cap2]+pulse[11, cap1])
					diff12 = np.abs(pulse[11, cap2] - pulse[11, cap1]-pulse[17, cap2]+pulse[17, cap1])
					diff13 = np.abs(pulse[11, cap2] - pulse[11, cap1]-pulse[23, cap2]+pulse[23, cap1])
					diff14 = np.abs(pulse[11, cap2] - pulse[11, cap1]-pulse[29, cap2]+pulse[29, cap1])
					diffsum  = diff01+diff02+diff03+diff04+diff11+diff12+diff13+diff14
					if max <diffsum:
						max = diffsum
						trigger_pos[e] = cap1 
		return trigger_pos
	#Reads a raw data file and saves a timebase calibration file.
	def create_timebase(self):
		
		sineData = Util.getDataRaw(self.measurement_config["timebase"]["input"])
		sineData = self.linearize_voltage(sineData) - 1.2/4096*0x800
		true_freq = self.measurement_config["timebase"]["true_freq"]#Frequency of the signal source used for timebase measurement.
		nevents = self.measurement_config["timebase"]["nevents"]
		#Find trigger position
		
		trigger_pos = self.find_trigger_pos(sineData)
		ydata = sineData

		timebase = np.zeros((30, 256))
		for channel in self.measurement_config["timebase"]["channels"]:
			chTimeOffsets = []
			x = []
			y =[]
			for iCap in range(256):
				
				cap1 = iCap
				cap2 = (iCap+1)%256
				r = []
				for e in range(0,nevents):
					if abs(trigger_pos[e]-cap1)<15 or (trigger_pos[e]-15<0 and 256-cap1+trigger_pos[e]<15) or (cap1-15<0 and 256-trigger_pos[e]+cap1<15):
						continue
					else: 
						r.append(e)


				x.append(ydata[r,channel, cap2] + ydata[r,channel, cap1])
				y.append(ydata[r,channel, cap1] - ydata[r,channel, cap2])

				#y = y[(x > 1200) | (x < -1200)]
				#x = x[(x > 1200) | (x < -1200)]
				#y = y[(x > 4) | (x < 2.3)]
				#x = x[(x > 4) | (x < 2.3)]

				# Formulate and solve the least squares problem ||Ax - b ||^2
				A = np.column_stack([x[iCap]**2, x[iCap] * y[iCap], y[iCap]**2, x[iCap], y[iCap]])
				b = np.ones_like(x[iCap])
				fit = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
				#print(fit)

				try:
					a = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]+math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
					#print("a = %f"%a)
					b = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]-math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
					#print("b = %f"%b)

					dtij = math.atan(b/a)/(math.pi*true_freq)
					#print("dtij = %f ps"%(dtij*1e12))
					chTimeOffsets.append(dtij)
				except:
					chTimeOffsets.append(100.0e-12)
			timebase[channel] = np.array(chTimeOffsets)
			print(channel)
		
		for i in range(30):
			self.df.at[i, "times"] = timebase[i]
		self.save()
		return sineData, trigger_pos

	#This function can be used AFTER a timebase calibration to visualize the phase distribution between channels.
	def phase_dist_between_channel(self):
		sineData = Util.getDataRaw(self.measurement_config["timebase"]["input"])
		sineData = self.linearize_voltage(sineData) - 1.2/4096*0x800
		trigger_pos = self.find_trigger_pos(sineData)
		true_freq = self.measurement_config["timebase"]["true_freq"]
		nevents = self.measurement_config["timebase"]["nevents"]
		phase = np.zeros((nevents, 30))
		chi = []
		for event in range(nevents):
			chi.append(0)
			xdataList = np.zeros((30, 256))
			ydataList = np.zeros((30, 256))
			sineCurve = np.zeros((30, 256))
			for channel in [11, 17, 23,29]:
				ringTimeOffsets = np.concatenate((self.df.at[channel, "times"],self.df.at[channel, "times"],self.df.at[channel, "times"]), 0)
				xdata = np.cumsum(ringTimeOffsets[trigger_pos[event]+14:trigger_pos[event]+14+256])
				ydata = np.concatenate((sineData[event,channel,:],sineData[event,channel,:],sineData[event,channel,:]))[trigger_pos[event]+15:trigger_pos[event]+15+256]
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[50:100], ydata[50:100], p0=(1.0, 0.0, true_freq*math.pi*2, 0.0), bounds=((0.1, -2, (true_freq-1)*math.pi*2, -4),(2, 2, (true_freq+1)*math.pi*2,4)))
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[:128], ydata[:128], p0=popt, bounds=((0.1, -2, (true_freq-1)*math.pi*2, -4),(2, 2, (true_freq+1)*math.pi*2,4)))
				diff = (ydata - Util.sine(xdata, *popt))/7e-4#np.abs(popt[0])
				chi[event]+=(diff**2)[:128].sum()/128
				phase[event, channel] = popt[3]
				
				xdataList[channel, :] = xdata
				ydataList[channel, :] = ydata
				sineCurve[channel, :] = Util.sine(xdata, *popt)
			if trigger_pos[event] < 120 and chi[event]>50000:
				self.plot(xdataList, ydataList)
				self.plot(xdataList, sineCurve)
		plt.title("Time offset and voltage calibration")
		plt.xlabel("Chi Squared")
		plt.ylabel("Events(total=1e5)")
		plt.hist(chi, bins="fd")
		
		phase2 = np.array([phase[x, :] for x in range(nevents) if trigger_pos[x] < 120]) #remove events that are too close to the wraparound.
		
		
		fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)
		ax1.hist(Util.wrap(phase2[:,17] - phase2[:,11])/true_freq/math.pi/2e-12, bins="fd")
		ax2.hist(Util.wrap(phase2[:,23] - phase2[:,11])/true_freq/math.pi/2e-12, bins="fd")
		ax3.hist(Util.wrap(phase2[:,29] - phase2[:,11])/true_freq/math.pi/2e-12, bins="fd")
		ax4.hist(Util.wrap(phase2[:,23] - phase2[:,17])/true_freq/math.pi/2e-12, bins="fd")
		ax5.hist(Util.wrap(phase2[:,29] - phase2[:,17])/true_freq/math.pi/2e-12, bins="fd")
		ax6.hist(Util.wrap(phase2[:,29] - phase2[:,23])/true_freq/math.pi/2e-12, bins="fd")
		
		plt.xlabel("phase diff(ps)")
		plt.ylabel(str(len(phase2))+" events")
		fig.tight_layout()
		plt.show()
		heatmap, xedges, yedges = np.histogram2d(Util.wrap(phase[:,29] - phase[:,23])/true_freq/math.pi/2e-12, chi, bins=(100,100))
		plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower", aspect="auto")
		plt.show()
		heatmap, xedges, yedges = np.histogram2d(Util.wrap(phase[:,29] - phase[:,17])/true_freq/math.pi/2e-12, chi, bins=(100,100))
		plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower", aspect="auto")
		plt.show()
		heatmap, xedges, yedges = np.histogram2d(Util.wrap(phase[:,29] - phase[:,11])/true_freq/math.pi/2e-12, chi, bins=(100,100))
		plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower", aspect="auto")
		plt.show()


	def plot(self, xdata, ydata):
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time (ns)")
		plt.ylabel("Voltage (V)")
		for channel in [11, 17, 23,29]:
			plt.plot(xdata[channel], ydata[channel])
		plt.legend(loc="lower right")
		plt.show()


	def savitzky_golay(y, window_size, order, deriv=0, rate=1):
		from math import factorial
		
		try:
			window_size = np.abs(int(window_size))
			order = np.abs(int(order))
		except(ValueError):
			raise ValueError("window_size and order have to be of type int")
		if window_size % 2 != 1 or window_size < 1:
			raise TypeError("window_size size must be a positive odd number")
		if window_size < order + 2:
			raise TypeError("window_size is too small for the polynomials order")
		order_range = range(order+1)
		half_window = (window_size -1) // 2
		# precompute coefficients
		b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
		m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
		# pad the signal at the extremes with
		# values taken from the signal itself
		firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
		lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
		y = np.concatenate((firstvals, y, lastvals))
		return np.convolve( m[::-1], y, mode='valid')
	

	def lineraize_wrap(f, val):
		try:
			return f(val)
		except(ValueError):
			if val < 2000:
				return 0
			else:
				return 3.3
	def linearize_voltage(self, sineData):
		x_vals = [i for i in range(self.measurement_config["voltage_curve"]["start"], self.measurement_config["voltage_curve"]["end"], self.measurement_config["voltage_curve"]["step"])]
		refVoltage = np.array([(float(i)/(2**12))*1.2 for i in x_vals])
		vlineraize_wrap = np.vectorize(Util.lineraize_wrap)
				
		voltageLin = []
		for j in range(0, 30):
			voltageLin.append([])
			for i in range(0, 256):
				meanList = Util.savitzky_golay(self.df.loc[j, "voltage_count_curves"][i, :, 1], 41, 2)
				#meanList[meanList<0] = 0
				voltageLin[j].append(scipy.interpolate.interp1d(meanList, refVoltage))
		#xv = np.array([acd for acd in range(4096)])
		#yv = vlineraize_wrap(voltageLin[0][0], xv)

		linDat = np.zeros_like(sineData, float)
		for j in range(0, 30):
			for i in range(0, 256):
				linDat[:,j,i] = Util.lineraize_wrap(voltageLin[j][i], sineData[:,j,i])
		return linDat
		
			

if __name__ == "__main__":
	try:
		a = sys.argv[1]
	except(IndexError):
		a = None
	ut = Util(a)
	#ut.create_voltage_curve()
	#ut.create_timebase()
	ut.phase_dist_between_channel()




