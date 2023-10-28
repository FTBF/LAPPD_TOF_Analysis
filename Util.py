import math
from textwrap import fill
import numpy as np
from datetime import datetime
import yaml
import bitstruct as bitstruct
import scipy
import scipy.optimize
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt
import multiprocessing as mp
import os.path
import sys
import uproot
from datetime import date
#Util Class is used for generating board calibration files and various measurements that are not directly used in TOF analysis.
#1. Voltage curve calibration file.
#2. Timebase calibration file.
#3. Phase distribution between channels.
OLD_FORMAT = False
DEBUG = False
VERBOSE = False
class Util:
	def __init__(self, board_config=None):
		self.measurement_config = Util.load_config(board_config)
		if(os.path.isfile(self.measurement_config["calibration_file"])):
			in_file = uproot.open(self.measurement_config["calibration_file"])
			
			self.voltage_df = np.reshape(in_file["config_tree"]["voltage_count_curves"].array(library="np"), (30,256,256, 2))
			self.time_df = np.reshape(in_file["config_tree"]["time_offsets"].array(library="np"), (30,256))
			
			print("Calibration file loaded.")
		else:
			print("Calibration file not found, creating new calibration file.")
			self.voltage_df = np.zeros((30,256,256, 2), dtype=np.float64)
			self.time_df = np.zeros((30,256), dtype=np.float64)
		self.trigger_pos = []
	def sine(x, A, B, omega, phi):
		return A * np.sin(omega*x + phi) + B

	#Just a sine function, with a coefficient array.
	def sineS(x, coefs):
		A = coefs[:, 0]
		B = coefs[:, 1]
		omega = coefs[:, 2]
		phi = coefs[:, 3]
		return np.transpose(A * np.sin(omega*np.transpose(x) + phi) + B)

	#Receives a radian and returns the value wrapped to [-pi, pi]
	def wrap(x):
		return np.remainder(x+math.pi, 2*math.pi)-math.pi
	def getDataRawOld(fname):
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

	def getDataRaw(fnames):
		if(OLD_FORMAT):
			return None, None, Util.getDataRawOld(fnames[0])
		N64BWORDS = 1440
		#N64BWORDS = 1536
		times_320 = []
		times = []
		data = []
		format_time320=bitstruct.compile("u64")
		format_time=bitstruct.compile("u64")
		format_header=bitstruct.compile("u16p48")
		format_accheader=bitstruct.compile("u56u8")
		format=bitstruct.compile("u12"*(256*30))
		#format=bitstruct.compile("p4u12u12u12u12u12"*(256*int(30/5)))
		swapformat="8"*(N64BWORDS)
		for fname in fnames:
			print(fname)
			with open(fname, "rb") as f:
				line = f.read((1+4+N64BWORDS)*8)
				lnum = 0
				while len(line) == (1+4+N64BWORDS)*8:
					acc_header = format_accheader.unpack(bitstruct.byteswap("8", line[0*8:1*8]))
					header = format_header.unpack(bitstruct.byteswap("8", line[1*8:2*8]))
					lnum += 1
					if acc_header[0] != 0x123456789abcde or header[0] != 0xac9c:
						#print("CORRUPT EVENT!!! ", lnum, "%x"%acc_header[0], "%x"%header[0])
						line = f.read((1+4+N64BWORDS)*8)
						continue
					times_320.extend(format_time320.unpack(bitstruct.byteswap("8", line[2*8:3*8])))
					times.extend(format_time320.unpack(bitstruct.byteswap("8", line[3*8:4*8])))
					data.extend(format.unpack(bitstruct.byteswap(swapformat, line[5*8:])))
					line = f.read((1+4+N64BWORDS)*8)
		data = np.array(data)
		times = np.array(times)
		times_320 = np.array(times_320)
		return times_320.reshape([-1,]), times.reshape([-1,]), data.reshape([-1, 30, 256])


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
			print("No Util.py config provided, loading default configuration")
			return Util.load_config("configs/calib_measurement_config.yml")

		with open(fn, "r") as stream:
			try:
				return yaml.safe_load(stream)
			except Exception as exc:
				print("Had an exception while reading yaml file for util config: %s"%exc)

	def save(self):
		output_file = uproot.recreate(self.measurement_config["calibration_file"])#Overwrite the existing file, if any.
		top_level = {"voltage_count_curves":self.voltage_df, "time_offsets":self.time_df}
		output_file["config_tree"] = top_level
		print("Calibration file saved.")
	
	#Reads a series of raw data files and saves a voltage curve calibration file.
	def create_voltage_curve(self):
		voltage_curve = []
		tmp_peak = np.zeros((30,256))
		x_vals = [i for i in range(self.measurement_config["voltage_curve"]["start"], self.measurement_config["voltage_curve"]["end"], self.measurement_config["voltage_curve"]["step"])]
		for i in x_vals:
			pedData = Util.getDataRaw([self.measurement_config["voltage_curve"]["prefix"]+str(i)+self.measurement_config["voltage_curve"]["suffix"]])[2]
			tmp_peak = np.maximum(tmp_peak, pedData.mean(0))#Force the curve to be monotonic.
			voltage_curve.append([np.full((30,256), i), tmp_peak])
			#voltage_curve.append([np.full((30,256), i), np.full((30,256), i)])
			print(i)
		#voltage_curve is a list of size (# of measurement points), each measurement point is a 2x30(ch)x256(# of capacitors) array of voltage values.
		voltage_curve = np.array(voltage_curve, dtype=np.float64).transpose(2,3,0,1) #Transpose so that the index order matches with voltage_df.
		self.voltage_df[:] = voltage_curve#Keep the address to the array the same so that TTree can read it.
		if(VERBOSE):
			for channel in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]:
				plt.plot(np.transpose(voltage_curve[channel, :,:,0]), np.transpose(voltage_curve[channel, :,:,1]))
				plt.xlabel("reference voltage (V)")
				plt.ylabel("measured ADC")
				plt.show()

		self.save()
	#Deprecated
	def find_trigger_pos2(self, sineData):
		nevents = self.measurement_config["timebase"]["nevents"]
		trigger_pos = [0]*nevents
		if not DEBUG:
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
	def find_trigger_pos(self, sineData):
		nevents = self.measurement_config["timebase"]["nevents"]
		true_freq = self.measurement_config["timebase"]["true_freq"]#Frequency of the signal source used for timebase measurement.
		ch = self.measurement_config["timebase"]["channels"][0]#In sake of simplicity, only use the first channel specified in the config file.	
		trigger_pos = [0]*nevents
		
		window = 16
		for e in range(0,nevents):
			pulse = np.concatenate((sineData[e,ch,:],sineData[e,ch,:]), 0)
			max = 0
			peak = np.max(pulse)
			
			for iCap in range(0, 256, 256//window):
				#Find the most indifferentiable point of the pulse.
				cap1 = iCap
				cap2 = iCap+window
				xdata = []
				for i in range(cap1, cap2):
					if i==256: 
						xdata.append(6e-10)#append approximate wraparound timebase. Hardware dependent.
					else: 
						xdata.append(1e-10)#append approximate timebase. Hardware dependent.
				xdata = np.cumsum(np.array(xdata))
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata, pulse[cap1:cap2], p0=(peak, 0.0, true_freq*math.pi*2, 0.0), bounds=((peak*0.5, -0.1, (true_freq*0.8)*math.pi*2, -4),(peak*1.1, 0.1, (true_freq*1.2)*math.pi*2,4)))
				diff = ((pulse[cap1:cap2] - Util.sine(xdata, *popt))**2).sum()
				if max <diff:
					max = diff
					trigger_pos[e] = cap1
			xdata = []
			a = trigger_pos[e]+window*3//2
			b = trigger_pos[e]+256-window//2
			for i in range(a,b):
				if i==256: 
					xdata.append(6e-10)#append approximate wraparound timebase. Hardware dependent.
				else: 
					xdata.append(1e-10)#append approximate timebase. Hardware dependent.
			xdata = np.cumsum(np.array(xdata))
			try:
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata, pulse[a:b], p0=(peak, 0.0, true_freq*math.pi*2, 0.0), bounds=((peak*0.5, -0.1, (true_freq*0.8)*math.pi*2, -4),(peak*1.1, 0.1, (true_freq*1.2)*math.pi*2,4)), max_nfev=50)
				trigger_pos[e] = (a + np.argmax(pulse[a:b] - Util.sine(xdata, *popt)))%256
			except(RuntimeError):
				pass
					
		plt.hist(trigger_pos, 256)
		plt.xlabel("trigger_pos")
		plt.show()
		return trigger_pos
	#Reads a raw data file and saves a timebase calibration file.
	def create_timebase_simple(self):
		sineData = Util.getDataRaw([self.measurement_config["timebase"]["input"]])[2]
		sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
		true_freq = self.measurement_config["timebase"]["true_freq"]#Frequency of the signal source used for timebase measurement.
		nevents = self.measurement_config["timebase"]["nevents"]
		#Find trigger position
		
		trigger_pos = self.find_trigger_pos(sineData)
		ydata = sineData

		timebase = np.zeros((30, 256))
		for channel in self.measurement_config["timebase"]["channels"]:
			chTimeOffsets = []
			chTimeStdevs = []
			x = []
			y =[]
			for iCap in range(256):
				
				cap1 = iCap
				cap2 =  0 if iCap==255 else min(iCap+1, 255)
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
				out = np.linalg.lstsq(A, b, rcond=None)
				fit = out[0].squeeze()

				try:
					a = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]+math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
					#print("a = %f"%a)
					b = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]-math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
					#print("b = %f"%b)

					dtij = math.atan(b/a)/(math.pi*true_freq)
					#print("dtij = %f ps"%(dtij*1e12))
					chTimeOffsets.append(dtij)
					chTimeStdevs.append(np.sqrt(out[1][0]))
					#chTimeOffsets.append(2.5e-8/256)
					plt.title("Channel %d, cap %d vs cap %d, t0 %d[ps]"%(channel, cap1, cap2, dtij*1e12))
					plt.scatter(x[iCap], y[iCap])
					# Plot the least squares ellipse
					x_coord = np.linspace(1.05*x[iCap].min(),1.05*x[iCap].max(),300)
					y_coord = np.linspace(1.05*y[iCap].min(),1.05*y[iCap].max(),300)
					X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
					Z_coord = fit[0] * (X_coord ** 2) + fit[1] * X_coord* Y_coord + fit[2] * Y_coord**2+ fit[3] * X_coord+ fit[4] * Y_coord
					plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
					plt.savefig("plots/timebase_i+9/channel%d_cap%d.png"%(channel, iCap))
					plt.close()
				except:
					print("Error in timebase calculation")
			timebase[channel] = np.array(chTimeOffsets)# + np.array(chTimeStdevs)*(2.45e-8 - np.sum(chTimeOffsets))/np.sum(chTimeStdevs)#Timebase normalization
			arr = np.zeros((256,256))
			for i in range(255):
				for j in range(255):
					if(j-i>=0 and j-i<1):
						arr[i,j] = 1
			arr[255,255] = 1
			timebase[channel] = np.dot(np.linalg.inv(arr),timebase[channel])

			plt.step(np.linspace(0, 255, 256), timebase[channel])
			plt.ylabel("time offset (s), total = %f ns"%(np.sum(timebase[channel])*1e9))
			plt.xlabel("timesample")
			plt.close()
			print(timebase[channel])
			print(sum(timebase[channel]))
		
		self.time_df[:] = timebase#Keep the address to the array the same so that TTree can read it.
		self.save()
		return sineData, trigger_pos
#Reads a raw data file and saves a timebase calibration file.
	def create_timebase_weighted(self):
		timebase = np.zeros((30, 256), dtype=np.float64)
		for channel in self.measurement_config["timebase"]["channels"]:
			sineData = Util.getDataRaw([self.measurement_config["timebase"]["prefix"]+str(channel)+self.measurement_config["timebase"]["suffix"]])[2]
			sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
			true_freq = self.measurement_config["timebase"]["true_freq"]#Frequency of the signal source used for timebase measurement.
			nevents = self.measurement_config["timebase"]["nevents"]
			ydata = sineData
			ydata2 = np.concatenate((ydata, ydata, ydata), axis=2)
			a_matrix = []
			y_matrix = []
			w_matrix = []
			for diff in [1, 2, 3, 4, 5, 6]:
				chTimeOffsetMatrix = []
				chTimeVarMatrix = []
				x = []
				y =[]
				coefs = []
				for iCap in range(256):
					cap1 = iCap+256
					cap2 = iCap+256+diff
					if(diff>1 and iCap+diff>255):continue
					x.append(ydata2[:,channel, cap2] + ydata2[:,channel, cap1])
					y.append(ydata2[:,channel, cap1] - ydata2[:,channel, cap2])
					# Formulate and solve the least squares problem ||Ax - b ||^2
					
					A = np.column_stack([x[iCap]**2, x[iCap] * y[iCap], y[iCap]**2, x[iCap], y[iCap]])
					b = np.ones_like(x[iCap])
					fit = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
					res = np.linalg.lstsq(A, b, rcond=None)[1]
					coefs.append(fit)
					#print(fit)

					try:  
						a = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]+math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
						#print("a = %f"%a)
						b = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]-math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
						#print("b = %f"%b)

						dtij = math.atan(b/a)/(math.pi*true_freq)
						#print("dtij = %f ps"%(dtij*1e12))
						chTimeOffsetMatrix.append(dtij)
						
					except:
						chTimeOffsetMatrix.append(100.0e-12)
					chTimeVarMatrix.append(res)
				vsize = 256-diff
				if(diff==1):vsize = 256
				arr = np.zeros((vsize,256))
				for i in range(vsize):
					for j in range(256):#Do not include wraparound terms for diff>1
						if(j-i>=0 and j-i<diff):
							arr[i,j] = 1
				a_matrix.append(arr)
				y_matrix.append(np.array(chTimeOffsetMatrix))
				w_matrix.append(np.array(chTimeVarMatrix))
			timebase[channel] = np.linalg.lstsq(np.divide(np.concatenate(a_matrix, axis=0),np.concatenate(w_matrix, axis=0)), np.divide(np.concatenate(y_matrix, axis=None),np.concatenate(w_matrix, axis=0).squeeze()), rcond=None)[0].squeeze()
		self.time_df[:] = timebase#Keep the address to the array the same so that TTree can read it.
		self.save()
		return sineData
	#Reads a raw data file and saves a timebase calibration file.
	def create_timebase_first_order(self):
		
		sineData = Util.getDataRaw([self.measurement_config["timebase"]["input"]])[2]
		sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
		true_freq = self.measurement_config["timebase"]["true_freq"]#Frequency of the signal source used for timebase measurement.
		nevents = self.measurement_config["timebase"]["nevents"]
		#Find trigger position0
		#trigger_pos = self.find_trigger_pos2(sineData)
		ydata = sineData

		timebase = np.zeros((30, 256))
		timebase_1 = np.zeros((30, 256))
		for channel in self.measurement_config["timebase"]["channels"]:
			chTimeOffsets = []
			chTimeStdevs = []
			x = []
			y =[]
			for iCap in range(256):
				
				cap1 = iCap
				cap2 = (iCap+1)%256
				r = []
				for e in range(0,nevents):
				#	if abs(trigger_pos[e]-cap1)<15 or (trigger_pos[e]-15<0 and 256-cap1+trigger_pos[e]<15) or (cap1-15<0 and 256-trigger_pos[e]+cap1<15):
				#		continue
				#	else: 
						r.append(e)


				x.append(ydata[r,channel, cap1])
				y.append(ydata[r,channel, cap1] - ydata[r,channel, cap2])
				#indarray = np.argsort(x[iCap])
				#ypeak_minus = np.median(y[iCap][:indarray[nevents//100]])
				#ypeak_plus = np.median(y[iCap][-indarray[nevents//100]:])
				#xpeak_minus = np.median(x[iCap][indarray[:nevents//100]])
				#xpeak_plus = np.median(x[iCap][indarray[-nevents//100:]])

				# Remove slope and offset, peakwise
				#y[iCap] = y[iCap] - (ypeak_plus + ypeak_minus)/2 - x[iCap]*(ypeak_plus - ypeak_minus)/2 / (xpeak_plus - xpeak_minus)

				# Formulate and solve the least squares problem ||Ax - b ||^2
				A = np.column_stack([x[iCap]**2+x[iCap]*y[iCap], y[iCap]**2, y[iCap], x[iCap]*y[iCap]**2+x[iCap]**2*y[iCap]])#y[iCap]**3])
				b = np.ones_like(x[iCap])
				out = np.linalg.lstsq(A, b, rcond=None)
				fit = out[0].squeeze()

				try:
					t0 = math.asin(math.sqrt(fit[0] / fit[1])/2) / true_freq /math.pi
					t1 = fit[2] / (fit[0] / fit[1]-2) * (math.sqrt(fit[0] / fit[1])/2) * math.sqrt(1-fit[0]/fit[1]/4) / true_freq / math.pi
					chTimeOffsets.append([t0, 0])
					

					#print("dtij = %f ps"%(dtij*1e12))
					#chTimeOffsets.append(2.5e-8/256)
					plt.title("Channel %d, cap %d, t0 %d[ps], t1 %d[ps/V]"%(channel, iCap, t0*1e12, t1*1e12))
					plt.scatter(x[iCap], y[iCap])
					# Plot the least squares ellipse
					x_coord = np.linspace(1.05*x[iCap].min(),1.05*x[iCap].max(),300)
					y_coord = np.linspace(1.05*y[iCap].min(),1.05*y[iCap].max(),300)
					X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
					Z_coord = fit[0] * (X_coord ** 2+X_coord*Y_coord) + fit[1] * Y_coord**2 + fit[2] * Y_coord + fit[3] * (X_coord * Y_coord**2 + X_coord**2 * Y_coord)
					plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
					plt.savefig("plots/timebase/channel%d_cap%d.png"%(channel, iCap))
					plt.close()
					
				except:
					print("Error in timebase calculation")
			timebase[channel] = np.array(chTimeOffsets)[:, 0]
			timebase_1[channel] = np.array(chTimeOffsets)[:, 1]
			plt.step(np.linspace(0, 255, 256), timebase[channel])
			plt.ylabel("time offset (s), total = %f ns"%(np.sum(timebase[channel])*1e9))
			plt.xlabel("timesample")
			plt.show()
			print(channel)

		#PHASE JITTER CALC
		phase = np.zeros((nevents, 30))
		freq = np.zeros((nevents, 30))
		chi = []
		for event in range(nevents):
			chi.append(0)
			xdataList = np.zeros((30, 256))
			ydataList = np.zeros((30, 256))
			sineCurve = np.zeros((30, 256))
			for channel in [11, 23]:
				
				ydata = np.concatenate((sineData[event,channel,:],sineData[event,channel,:],sineData[event,channel,:]))[trigger_pos[event]+15:trigger_pos[event]+15+256]
				ydot = np.gradient(sineData[event,channel,:], axis=0)
				ringTimeOffsets = np.concatenate((timebase[channel]+ydot*timebase_1[channel],timebase[channel]+ydot*timebase_1[channel],timebase[channel]+ydot*timebase_1[channel]), 0)
				xdata = np.cumsum(ringTimeOffsets[trigger_pos[event]+14:trigger_pos[event]+14+256])+np.sum(ringTimeOffsets[0:trigger_pos[event]+14])
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[50:80], ydata[50:80], p0=(0.5, 0.0, true_freq*math.pi*2, 0.0), bounds=((0.1, -0.1, (true_freq*0.9999)*math.pi*2, -4),(1, 0.1, (true_freq*1.0001)*math.pi*2,4)))
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[:128], ydata[:128], p0=popt, bounds=((0.1, -0.1, (true_freq*0.9999)*math.pi*2, -4),(1, 0.1, (true_freq*1.0001)*math.pi*2,4)))
				diff = (ydata - Util.sine(xdata, *popt))/7e-4#np.abs(popt[0])
				chi[event]+=(diff**2)[:128].sum()/128/4
				phase[event, channel] = popt[3]
				freq[event, channel] = popt[2]/2/math.pi
				xdataList[channel, :] = xdata
				ydataList[channel, :] = ydata
				sineCurve[channel, :] = Util.sine(xdata, *popt)
			#if trigger_pos[event] < 120:
			#	self.plot(xdataList, ydataList)
			#	self.plot(xdataList, sineCurve)
		plt.title("Time offset and voltage calibration")
		plt.xlabel("Chi Squared")
		plt.ylabel("Events(total="+str(nevents)+")")
		plt.hist(chi, bins="fd")
		plt.show()
		
		phase2 = np.array([phase[x, :] for x in range(nevents) if chi[x] < 3000]) #remove events that are too close to the wraparound.
		#if trigger_pos[x] < 120
		
		fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 1)
		ax1.hist(Util.wrap(phase2[:,23] - phase2[:,11])/true_freq/math.pi/2e-12, bins="fd")
		
		plt.xlabel("phase diff(ps)")
		plt.ylabel(str(len(phase2))+" events")
		fig.tight_layout()
		plt.show()
		heatmap, xedges, yedges = np.histogram2d(Util.wrap(phase2[:,11] - phase2[:,23])/true_freq/math.pi/2e-12, chi, bins=100)
		plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower", aspect="auto")
		plt.show()
		heatmap, xedges, yedges = np.histogram2d(np.average(freq, 1)*7.5, chi, bins=100)
		plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower", aspect="auto")
		plt.xlabel("frequency(Hz)")
		plt.ylabel("chi squared")
		plt.show()
		#for i in range(30):
		#	self.df.at[i, "times"] = timebase[i]
		#self.save()
		#return sineData, trigger_pos

	#This function can be used AFTER a timebase calibration to visualize the phase distribution between channels.
	def phase_dist_between_channel(self):
		sineData = Util.getDataRaw([self.measurement_config["timebase"]["input"]])[2]
		if not DEBUG:
			sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
		else:
			sineData = 1.2/4096*sineData - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
		trigger_pos = self.find_trigger_pos2(sineData)
		true_freq = self.measurement_config["timebase"]["true_freq"]
		nevents = self.measurement_config["timebase"]["nevents"]
		phase = np.zeros((nevents, 30))
		freq = np.zeros((nevents, 30))
		chi = []
		for event in range(nevents):
			chi.append(0)
			xdataList = np.zeros((30, 256))
			ydataList = np.zeros((30, 256))
			sineCurve = np.zeros((30, 256))
			for channel in [11, 17, 23,29]:
				ringTimeOffsets = np.concatenate((self.df.at[channel, "times"],self.df.at[channel, "times"],self.df.at[channel, "times"]), 0)
				xdata = np.cumsum(ringTimeOffsets[trigger_pos[event]+14:trigger_pos[event]+14+256])+np.sum(ringTimeOffsets[0:trigger_pos[event]+14])
				ydata = np.concatenate((sineData[event,channel,:],sineData[event,channel,:],sineData[event,channel,:]))[trigger_pos[event]+15:trigger_pos[event]+15+256]
				popt, pcov = scipy.optimize.curve_fit(lambda x,a,b,phi: Util.sine(x,a,b,2*math.pi*true_freq, phi), xdata[50:80], ydata[50:80], p0=(0.5, 0.0, 0.0), bounds=((0.1, -0.1, -4),(1, 0.1,4)))
				popt, pcov = scipy.optimize.curve_fit(lambda x,a,b,phi: Util.sine(x,a,b,2*math.pi*true_freq, phi), xdata[:128], ydata[:128], p0=popt, bounds=((0.1, -0.1, -4),(1, 0.1,4)))
				diff = (ydata - Util.sine(xdata, popt[0],popt[1],2*math.pi*true_freq,popt[2]))/7e-4#np.abs(popt[0])
				chi[event]+=(diff**2)[:128].sum()/128/4
				phase[event, channel] = popt[2]
				freq[event, channel] = true_freq #popt[1]/2/math.pi
				xdataList[channel, :] = xdata
				ydataList[channel, :] = ydata
				sineCurve[channel, :] = Util.sine(xdata, popt[0],popt[1],2*math.pi*true_freq,popt[2])
				if(DEBUG and event%100 == 0):
					plt.title("Sample Event")
					plt.xlabel("xdata, variance="+str(np.var(xdata))+",trigger_pos="+str(trigger_pos[event]))
					plt.ylabel("ydata, phase="+str(popt[2])+", amplitude="+str(popt[0])+", offset="+str(popt[1]))
					plt.plot(xdata, Util.sine(xdata, popt[0],popt[1],2*math.pi*true_freq,popt[2]), label="sine fit")
					plt.scatter(xdata, ydata)
					plt.show()
			#if trigger_pos[event] < 120:
			#	self.plot(xdataList, ydataList)
			#	self.plot(xdataList, sineCurve)
		plt.title("Time offset and voltage calibration")
		plt.xlabel("Chi Squared")
		plt.ylabel("Events(total="+str(nevents)+")")
		plt.hist(chi, bins="fd")
		plt.show()
		
		# plt.title("Time offset and voltage calibration")
		# plt.xlabel("Frequency")
		# plt.ylabel("Events(total="+str(nevents)+")")
		# plt.hist(freq[:,11], bins="fd")
		# plt.show()
		
		phase2 = np.array([phase[x, :] for x in range(nevents) if chi[x] < 65000]) #remove events that are too close to the wraparound.
		#if trigger_pos[x] < 120
		
		fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)
		ax1.hist(Util.wrap(phase2[:,17])/true_freq/math.pi/2e-12, bins="fd")
		ax2.hist(Util.wrap(phase2[:,23])/true_freq/math.pi/2e-12, bins="fd")
		ax3.hist(Util.wrap(phase2[:,29])/true_freq/math.pi/2e-12, bins="fd")
		ax4.hist(Util.wrap(phase2[:,23] - phase2[:,17])/true_freq/math.pi/2e-12, bins="fd")
		ax5.hist(Util.wrap(phase2[:,29] - phase2[:,17])/true_freq/math.pi/2e-12, bins="fd")
		ax6.hist(Util.wrap(phase2[:,29] - phase2[:,23])/true_freq/math.pi/2e-12, bins="fd")
		
		plt.xlabel("phase diff(ps)")
		plt.ylabel(str(len(phase2))+" events")
		fig.tight_layout()
		plt.show()

		plt.hist(Util.wrap(phase2[:,29] - phase2[:,23])/true_freq/math.pi/2e-12, bins="fd")
		mean = np.mean(Util.wrap(phase2[:,29] - phase2[:,23])/true_freq/math.pi/2e-12)
		std = np.std(Util.wrap(phase2[:,29] - phase2[:,23])/true_freq/math.pi/2e-12)
		plt.xlabel(str(mean)+" +/- "+str(std)+" ps")
		plt.ylabel(str(len(phase2))+" events")
		plt.show()

		heatmap, xedges, yedges = np.histogram2d(Util.wrap(phase2[:,29] - phase2[:,23])/true_freq/math.pi/2e-12, chi, bins=100)
		plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower", aspect="auto")
		plt.show()
		heatmap, xedges, yedges = np.histogram2d(np.average(freq, 1)*7.5, chi, bins=100)
		plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower", aspect="auto")
		plt.xlabel("frequency(Hz)")
		plt.ylabel("chi squared")
		plt.show()
	def square_norm(t, fun_list, event):
		return scipy.integrate.quad(lambda x:(fun_list[event][0](x+t) - fun_list[event][1](t))**2 , 0, 100e-9, epsrel=1e-2)[0]
#This function can be used AFTER a timebase calibration to visualize the phase distribution between channels.
	def phase_dist_square_norm(self):
		sineData = Util.getDataRaw([self.measurement_config["timebase"]["input"]])[2]
		sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
		trigger_pos = self.find_trigger_pos2(sineData)
		nevents = self.measurement_config["timebase"]["nevents"]
		fun_list = []
		xdata_list = []
		ydata_list = []
		t_list = []
		print(self.df.at[11, "times"])
		for event in range(nevents):
			fun_list.append([])
			xdata_list.append([])
			ydata_list.append([])
			for channel in [11, 17, 23,29]:
				ringTimeOffsets = np.concatenate((self.df.at[channel, "times"],self.df.at[channel, "times"],self.df.at[channel, "times"]), 0)
				xdata = np.cumsum(ringTimeOffsets[trigger_pos[event]+14:trigger_pos[event]+14+256])+np.sum(ringTimeOffsets[0:trigger_pos[event]+14])
				ydata = np.concatenate((sineData[event,channel,:],sineData[event,channel,:],sineData[event,channel,:]))[trigger_pos[event]+15:trigger_pos[event]+15+256]
				fun_list[event].append(scipy.interpolate.interp1d(xdata[:128], ydata[:128], bounds_error=False, fill_value=0))
				xdata_list[event].append(xdata)
				ydata_list[event].append(ydata)
				#fun_list[event].append(scipy.interpolate.interp1d(np.linspace(0, 100e-9, 100), np.linspace(0, 0, 100), bounds_error=False, fill_value=0))
			
			#t_list.append(scipy.optimize.brute(Util.square_norm, [slice(-3e-11, 3e-11, 1e-12)], workers=-1, args=(fun_list, event))[0])
			t = scipy.optimize.curve_fit(lambda x,t: fun_list[event][0](x+t), xdata_list[event][1][:128], ydata_list[event][1][:128], p0=0, bounds=(-3e-11, 3e-11))[0][0]
			t_list.append(scipy.optimize.curve_fit(lambda x,t: fun_list[event][0](x+t), xdata_list[event][1][:128], ydata_list[event][1][:128], p0=t, bounds=(-1e-10, 1e-10))[0][0])
			#print(str(event)+": "+str(t_list[event]))
		plt.scatter(xdata_list[50][0], ydata_list[50][0])
		plt.scatter(xdata_list[50][1], ydata_list[50][1])
		plt.show()
		plt.cla()
		print(t_list)
		plt.hist(t_list, bins='fd')
		plt.show()
		plt.cla()
		plt.hist(t_list, bins=np.linspace(-1e-11, 1e-11, 100))
		plt.show()
		plt.cla()
	
			
	def timebase_slope_1p_correction(self):
		sineData = Util.getDataRaw([self.measurement_config["timebase"]["input"]])[2]
		sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
		trigger_pos = self.find_trigger_pos2(sineData)
		trigger_cutoff = 200
		true_freq = self.measurement_config["timebase"]["true_freq"]
		nevents = self.measurement_config["timebase"]["nevents"]
		xdataList = np.zeros((nevents, 30, trigger_cutoff))
		ydataList = np.zeros((nevents, 30, trigger_cutoff))
		ydot = np.zeros((nevents, 30, 256))
		sineCurve = np.zeros((nevents, 30, trigger_cutoff))
		coefs = np.zeros((nevents, 30, 4))
		for event in range(nevents):
			for channel in [11]:
				ringTimeOffsets = np.concatenate((self.df.at[channel, "times"],self.df.at[channel, "times"],self.df.at[channel, "times"]), 0)
				xdata = np.cumsum(ringTimeOffsets[trigger_pos[event]+14:trigger_pos[event]+14+256])+np.sum(ringTimeOffsets[0:trigger_pos[event]+14])
				ydata = np.concatenate((sineData[event,channel,:],sineData[event,channel,:],sineData[event,channel,:]))[trigger_pos[event]+15:trigger_pos[event]+15+256]
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[50:80], ydata[50:80], p0=(0.5, 0.0, true_freq*math.pi*2, 0.0), bounds=((0.1, -0.1, (true_freq*0.9999)*math.pi*2, -4),(1, 0.1, (true_freq*1.0001)*math.pi*2,4)))
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[:128], ydata[:128], p0=popt, bounds=((0.1, -0.1, (true_freq*0.9)*math.pi*2, -4),(1, 0.1, (true_freq*1.1)*math.pi*2,4)))
				xdataList[event, channel, :] = xdata[:trigger_cutoff]
				ydataList[event, channel, :] = ydata[:trigger_cutoff]
				sineCurve[event, channel, :] = Util.sine(xdata[:trigger_cutoff], *popt)
				coefs[event, channel, :] = popt
		ydot = np.gradient(ydataList, axis=2)
		A = [-1e-10, 0]
		result = scipy.optimize.basinhopping(lambda x: np.sum((ydataList[:,channel] - Util.sineS(xdataList[:,channel]+ydot[:,channel]*x[0]+x[1], coefs[:, channel]))**2), A, disp=True, T=nevents * 1e-5, stepsize=1e-9, niter=1000)
		print(result)
		parameters = result["x"]
		plt.plot(parameters)
		plt.show()
		event = 3
		#print(np.sum((ydataList[event] - Util.sine(xdataList[event]+ydot[event]*np.dot(reorder[event], result["x"])))**2, axis=1))
		self.plot(xdataList[event], ydataList[event])
		self.plot(xdataList[event]+ydot[event]*parameters[0]+parameters[1], ydataList[event])
		self.plot(xdataList[event], ydataList[event] - Util.sine(xdataList[event], *coefs[event, channel]))
		self.plot(xdataList[event]+ydot[event]*parameters[0]+parameters[1], ydataList[event] - Util.sine(xdataList[event]+ydot[event]*parameters[0]+parameters[1], *coefs[event, channel]))

		#PHASE JITTER CALC
		phase = np.zeros((nevents, 30))
		freq = np.zeros((nevents, 30))
		chi = []
		for event in range(nevents):
			chi.append(0)
			xdataList = np.zeros((30, 256))
			ydataList = np.zeros((30, 256))
			sineCurve = np.zeros((30, 256))
			for channel in [11, 17, 23,29]:
				ringTimeOffsets = np.concatenate((self.df.at[channel, "times"],self.df.at[channel, "times"],self.df.at[channel, "times"]), 0)
				xdata = np.cumsum(ringTimeOffsets[trigger_pos[event]+14:trigger_pos[event]+14+256])+np.sum(ringTimeOffsets[0:trigger_pos[event]+14])
				ydata = np.concatenate((sineData[event,channel,:],sineData[event,channel,:],sineData[event,channel,:]))[trigger_pos[event]+15:trigger_pos[event]+15+256]
				ydot = np.gradient(ydata, axis=0)
				xdata = xdata + ydot * A[0] + A[1]
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[50:80], ydata[50:80], p0=(0.5, 0.0, true_freq*math.pi*2, 0.0), bounds=((0.1, -0.1, (true_freq*0.9999)*math.pi*2, -4),(1, 0.1, (true_freq*1.0001)*math.pi*2,4)))
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[:128], ydata[:128], p0=popt, bounds=((0.1, -0.1, (true_freq*0.9999)*math.pi*2, -4),(1, 0.1, (true_freq*1.0001)*math.pi*2,4)))
				diff = (ydata - Util.sine(xdata, *popt))/7e-4#np.abs(popt[0])
				chi[event]+=(diff**2)[:128].sum()/128/4
				phase[event, channel] = popt[3]
				freq[event, channel] = popt[2]/2/math.pi
				xdataList[channel, :] = xdata
				ydataList[channel, :] = ydata
				sineCurve[channel, :] = Util.sine(xdata, *popt)
			#if trigger_pos[event] < 120:
			#	self.plot(xdataList, ydataList)
			#	self.plot(xdataList, sineCurve)
		plt.title("Time offset and voltage calibration")
		plt.xlabel("Chi Squared")
		plt.ylabel("Events(total="+str(nevents)+")")
		plt.hist(chi, bins="fd")
		plt.show()
		
		plt.title("Time offset and voltage calibration")
		plt.xlabel("Frequency")
		plt.ylabel("Events(total="+str(nevents)+")")
		plt.hist(freq[:,11], bins="fd")
		plt.show()
		
		phase2 = np.array([phase[x, :] for x in range(nevents) if chi[x] < 3000]) #remove events that are too close to the wraparound.
		#if trigger_pos[x] < 120
		
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
		heatmap, xedges, yedges = np.histogram2d(Util.wrap(phase2[:,29] - phase2[:,23])/true_freq/math.pi/2e-12, chi, bins=100)
		plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower", aspect="auto")
		plt.show()
		heatmap, xedges, yedges = np.histogram2d(np.average(freq, 1)*7.5, chi, bins=100)
		plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower", aspect="auto")
		plt.xlabel("frequency(Hz)")
		plt.ylabel("chi squared")
		plt.show()
		pass
	
	def timebase_slope_correction(self):
		sineData = Util.getDataRaw([self.measurement_config["timebase"]["input"]])[2]
		sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
		trigger_pos = self.find_trigger_pos2(sineData)
		trigger_cutoff = 200
		true_freq = self.measurement_config["timebase"]["true_freq"]
		nevents = self.measurement_config["timebase"]["nevents"]
		phase = np.zeros((nevents, 30))
		freq = np.zeros((nevents, 30))
		chi = []
		xdataList = np.zeros((nevents, 30, trigger_cutoff))
		ydataList = np.zeros((nevents, 30, trigger_cutoff))
		ydot = np.zeros((nevents, 30, 256))
		sineCurve = np.zeros((nevents, 30, trigger_cutoff))
		coefs = np.zeros((nevents, 30, 4))
		for event in range(nevents):
			chi.append(0)
			for channel in [11]:
				ringTimeOffsets = np.concatenate((self.df.at[channel, "times"],self.df.at[channel, "times"],self.df.at[channel, "times"]), 0)
				xdata = np.cumsum(ringTimeOffsets[trigger_pos[event]+14:trigger_pos[event]+14+256])+np.sum(ringTimeOffsets[0:trigger_pos[event]+14])
				ydata = np.concatenate((sineData[event,channel,:],sineData[event,channel,:],sineData[event,channel,:]))[trigger_pos[event]+15:trigger_pos[event]+15+256]
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[50:80], ydata[50:80], p0=(0.5, 0.0, true_freq*math.pi*2, 0.0), bounds=((0.1, -0.1, (true_freq*0.9999)*math.pi*2, -4),(1, 0.1, (true_freq*1.0001)*math.pi*2,4)))
				popt, pcov = scipy.optimize.curve_fit(Util.sine, xdata[:128], ydata[:128], p0=popt, bounds=((0.1, -0.1, (true_freq*0.9)*math.pi*2, -4),(1, 0.1, (true_freq*1.1)*math.pi*2,4)))
				diff = (ydata - Util.sine(xdata, *popt))/7e-4#np.abs(popt[0])
				chi[event]+=(diff**2)[:128].sum()/128/4
				phase[event, channel] = popt[3]
				freq[event, channel] = popt[2]/2/math.pi
				xdataList[event, channel, :] = xdata[:trigger_cutoff]
				ydataList[event, channel, :] = ydata[:trigger_cutoff]
				sineCurve[event, channel, :] = Util.sine(xdata[:trigger_cutoff], *popt)
				coefs[event, channel, :] = popt
		ydot = np.gradient(ydataList, axis=2)
		"""
		B = -1e-10
		#result = scipy.optimize.least_squares(lambda x: np.sqrt(np.sum((ydataList[:,channel] - Util.sineS(xdataList[:,channel]+ydot[:,channel]*np.dot(reorder, x), coefs[:, channel]))**2)), A, verbose=2, x_scale="jac", xtol=1e-16)
		#result = scipy.optimize.basinhopping(lambda x: np.sum((ydataList[:,channel] - Util.sineS(xdataList[:,channel]+ydot[:,channel]*np.dot(reorder, x), coefs[:, channel]))**2), A, disp=True, T=nevents * 1e-1, stepsize=1e-9)
		#result = scipy.optimize.basinhopping(lambda x: np.sum((ydataList[:,channel] - Util.sineS(xdataList[:,channel]+ydot[:,channel]*np.dot(reorder, np.ones(256)*x), coefs[:, channel]))**2), B, disp=True, T=nevents * 1e-5, stepsize=1e-9, niter=100)
		#one parameter fit result : -1.0154391e-10
		#one parameter fit ratio : 0.360048 / 0.362656
		result = scipy.optimize.minimize(lambda x: np.sum((ydataList[:,channel] - Util.sineS(xdataList[:,channel]+ydot[:,channel]*np.dot(reorder, np.ones(256)*x*1e-12), coefs[:, channel]))**2), B, method="BFGS", options={"disp":True, "maxiter":1000})	
		
		print(result)
		parameters = np.ones(256)*result["x"]
		plt.plot(parameters)
		plt.show()
		event = 3
		#print(np.sum((ydataList[event] - Util.sine(xdataList[event]+ydot[event]*np.dot(reorder[event], result["x"])))**2, axis=1))
		self.plot(xdataList[event], ydataList[event])
		self.plot(xdataList[event]+ydot[event]*np.dot(reorder[event], parameters), ydataList[event])
		self.plot(xdataList[event], ydataList[event] - Util.sine(xdataList[event], *coefs[event, channel]))
		self.plot(xdataList[event]+ydot[event]*np.dot(reorder[event], parameters), ydataList[event] - Util.sine(xdataList[event]+ydot[event]*np.dot(reorder[event], parameters), *coefs[event, channel]))
		pass

		"""
		
		A = []
		parameters = []
		#Reorder the A, so that ydata, ydot, and xdata are aligned with the order of capacitors
		reorder = np.array([np.roll(np.identity(256), trigger_pos[i]+15, 1)[:trigger_cutoff] for i in range(nevents)])
		for iCap in range(256):
			A.append([])
			parameters.append([])
			for channel in [11]:
				A[iCap].append([])
				parameters[iCap].append([])
				for event in range(nevents):
					A[iCap][channel].append(ydot[event, channel, iCap])
				A[iCap][channel] = np.array(A[iCap][channel])
				parameters[iCap][channel] = np.linalg.lstsq(A[iCap][channel].reshape(-1, 1), ydataList[:, channel, iCap], rcond=None)[0][0]

		print(result)
		parameters = result["x"]
		plt.plot(parameters)
		plt.show()
		event = 3
		#print(np.sum((ydataList[event] - Util.sine(xdataList[event]+ydot[event]*np.dot(reorder[event], result["x"])))**2, axis=1))
		self.plot(xdataList[event], ydataList[event])
		self.plot(xdataList[event]+ydot[event]*np.dot(reorder[event], parameters), ydataList[event])
		self.plot(xdataList[event], ydataList[event] - Util.sine(xdataList[event], *coefs[event, channel]))
		self.plot(xdataList[event]+ydot[event]*np.dot(reorder[event], parameters), ydataList[event] - Util.sine(xdataList[event]+ydot[event]*np.dot(reorder[event], parameters), *coefs[event, channel]))
		pass





	def plot(self, xdata, ydata):
		plt.title("Outlying event")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [11]:
			plt.plot(xdata[channel], ydata[channel])
		plt.legend(loc="lower right")
		plt.show()
	def raw_plot(self):
		event = self.measurement_config["plot"]["event"]
		rawData = Util.getDataRaw([self.measurement_config["plot"]["input"]])[2]
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [0,1,2,3,4,5]:
			plt.plot(np.linspace(0, 255,256), rawData[event, channel, :], label=str(channel))
		plt.legend(loc="lower right")
		plt.show()
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [6,7,8,9,10,11]:
			plt.plot(np.linspace(0, 255,256), rawData[event, channel, :], label=str(channel))
		plt.legend(loc="lower right")
		plt.show()
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [12,13,14,15,16,17]:
			plt.plot(np.linspace(0, 255,256), rawData[event, channel, :], label=str(channel))
		plt.legend(loc="lower right")
		plt.show()
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [18,19,20,21,22,23]:
			plt.plot(np.linspace(0, 255,256), rawData[event, channel, :], label=str(channel))
		plt.legend(loc="lower right")
		plt.show()
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [24,25,26,27,28,29]:
			plt.plot(np.linspace(0, 255,256), rawData[event, channel, :], label=str(channel))
		plt.legend(loc="lower right")
		plt.show()
	def simple_plot(self):
		event = self.measurement_config["plot"]["event"]
		rawData = Util.getDataRaw([self.measurement_config["plot"]["input"]])[2]
		sineData = self.linearize_voltage(rawData) - 1.2/4096*self.measurement_config["plot"]["pedestal"]
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [0,1,2,3,4,5]:
			plt.plot(np.linspace(0, 255,256), sineData[event, channel, :], label=str(channel))
		plt.legend(loc="lower right")
		plt.show()
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [6,7,8,9,10,11]:
			plt.plot(np.linspace(0, 255,256), sineData[event, channel, :], label=str(channel))
		plt.legend(loc="lower right")
		plt.show()
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [12,13,14,15,16,17]:
			plt.plot(np.linspace(0, 255,256), sineData[event, channel, :], label=str(channel))
		plt.legend(loc="lower right")
		plt.show()
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [18,19,20,21,22,23]:
			plt.plot(np.linspace(0, 255,256), sineData[event, channel, :], label=str(channel))
		plt.legend(loc="lower right")
		plt.show()
		plt.title("Time offset and voltage calibration")
		plt.xlabel("time [s]")
		plt.ylabel("Voltage [V]")
		for channel in [24,25,26,27,28,29]:
			plt.plot(np.linspace(0, 255,256), sineData[event, channel, :], label=str(channel))
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
			return val/4096*1.2
			
	def linearize_voltage(self, sineData):
		x_vals = [i for i in range(self.measurement_config["voltage_curve"]["start"], self.measurement_config["voltage_curve"]["end"], self.measurement_config["voltage_curve"]["step"])]
		refVoltage = np.array([(float(i)/(2**12))*1.2 for i in x_vals])
		vlineraize_wrap = np.vectorize(Util.lineraize_wrap)
				
		voltageLin = []
		for j in range(0, 30):
			voltageLin.append([])
			for i in range(0, 256):
				meanList = Util.savitzky_golay(self.voltage_df[j, i, :, 1], 41, 2)
				meanList[meanList<0] = 0
				meanList[255] = 1.2 #fix the last value so that interpolation doesn't go out of bounds
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
	if 'o' in ut.measurement_config["tasks"]:
		OLD_FORMAT = True
	if 'D' in ut.measurement_config["tasks"]:
		DEBUG = True
	if 'V' in ut.measurement_config["tasks"]:
		VERBOSE = True
	if 'r' in ut.measurement_config["tasks"]:
		ut.raw_plot()
	if 'p' in ut.measurement_config["tasks"]:
		ut.simple_plot()
	if 'v' in ut.measurement_config["tasks"]:
		ut.create_voltage_curve()
	if 't' in ut.measurement_config["tasks"]:
		ut.create_timebase_simple()
	if 'w' in ut.measurement_config["tasks"]:
		ut.create_timebase_weighted()
	if 'T' in ut.measurement_config["tasks"]:
		ut.create_timebase_first_order()
	if 'j' in ut.measurement_config["tasks"]:
		ut.phase_dist_between_channel()
	if 'l' in ut.measurement_config["tasks"]:
		ut.timebase_slope_1p_correction()
	if 'n' in ut.measurement_config["tasks"]:
		ut.phase_dist_square_norm()




