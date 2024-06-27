import math
import random
from textwrap import fill
import numpy as np
from datetime import datetime
import yaml
import cbitstruct as bitstruct
import scipy
import scipy.optimize
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing as mp
import os.path
import sys
import uproot
from sigfig import round
from datetime import date
import glob
#Util Class is used for generating board calibration files and various measurements that are not directly used in TOF analysis.
#1. Voltage curve calibration file.
#2. Timebase calibration file.
#3. Phase distribution between channels.
OLD_FORMAT = False
DEBUG = False
VERBOSE = False
SAVE = True
WRAPAROUND_ONLY = False
ERRORBAR = False
VCC_NONLINEARITY = False
class Util:
	def __init__(self, board_config=None):
		self.measurement_config = Util.load_config(board_config)
		if os.path.isfile(self.measurement_config["calibration_file"]):
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

	def getDataRaw(fnames, startevent=0, batchsize=-1):
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
				f.read((1+4+N64BWORDS)*8*startevent)
				line = f.read((1+4+N64BWORDS)*8)
				lnum = startevent
				while len(line) == (1+4+N64BWORDS)*8:
					if batchsize > 0 and lnum >= startevent + batchsize:
						break
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
		data = np.array(data).reshape([-1,30,256])
		times = np.array(times).reshape([-1])
		times_320 = np.array(times_320).reshape([-1])

		return times_320, times, data


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
		chan = 0
		cap = 126
		ydata = np.array([], dtype=np.float64)
		xdata = np.array([], dtype=np.float64)
		for i in x_vals:
			pedData = Util.getDataRaw([self.measurement_config["voltage_curve"]["prefix"]+str(i)+self.measurement_config["voltage_curve"]["suffix"]])[2]
			chanCapPedData = pedData[:,chan,cap]
			# if i == 0xad0 or i == 0xae0:
			# 	fig, ax = plt.subplots()
			# 	ax.hist(chanCapPedData, bins=100)
			# 	plt.show()
			ydata = np.hstack([ydata, np.full_like(chanCapPedData, i*1.2/4096., dtype=np.float64)])
			xdata = np.hstack([xdata, chanCapPedData])
			tmp_peak = np.maximum(tmp_peak, pedData.mean(0))#Force the curve to be monotonic.
			voltage_curve.append([np.full((30,256), i), tmp_peak])
			#voltage_curve.append([np.full((30,256), i), np.full((30,256), i)])
			print(i)
		# fig, ax = plt.subplots()
		# _, _, _, image_mesh = ax.hist2d(xdata, ydata, bins=250)
		# fig.colorbar(image_mesh, ax=ax)
		# plt.show()
		#voltage_curve is a list of size (# of measurement points), each measurement point is a 2x30(ch)x256(# of capacitors) array of voltage values.
		voltage_curve = np.array(voltage_curve, dtype=np.float64).transpose(2,3,0,1) #Transpose so that the index order matches with voltage_df.
		# self.voltage_df[:] = voltage_curve#Keep the address to the array the same so that TTree can read it.
		if(VERBOSE):
			for ch in range(30):
				fig, ax = plt.subplots()
				ax.set_title(f'Channel {ch}, PSEC{11+ch//6}')
				for cap in range(256):
					ax.plot(voltage_curve[ch, cap, :, 1], voltage_curve[ch, cap, :, 0]*1.2/4096.)
					# ax.plot(voltage_curve[ch, cap, :, 0], voltage_curve[ch, cap, :, 1]*1.2/4096.)
				ax.set_xlabel('Measured ADC count')
				ax.set_ylabel('DAC Voltage (V)')
				ax.xaxis.set_ticks_position('both')
				ax.yaxis.set_ticks_position('both')
				plt.minorticks_on()
				plt.show()
		if(SAVE):
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
		times320, _, sineData = Util.getDataRaw([self.measurement_config["timebase"]["input"]])[2]
		sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
		true_freq = self.measurement_config["timebase"]["true_freq"]#Frequency of the signal source used for timebase measurement.
		nevents = self.measurement_config["timebase"]["nevents"]
		#Find trigger position
		
		trigger_pos = ((times320 % 8))*32-16
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
	
	def plot_ellipse(self):
		
		file_path = self.measurement_config["plot"]["input"]
		channel = self.measurement_config["plot"]["channel"]
		iCap = self.measurement_config["plot"]["cap"]

		times320, _, sineData = Util.getDataRaw([file_path])
		trigger_pos = (((times320+2+2)%8)*32-16)%256
		sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
		true_freq = self.measurement_config["timebase"]["true_freq"]#Frequency of the signal source used for timebase measurement.
		ydata = sineData
		nevents = ydata.shape[0]
		ydata2 = np.concatenate((ydata, ydata, ydata), axis=2)
			

		x = []
		y =[]
		diff = 1
		cap1 = iCap+256
		cap2 = iCap+256+diff
		if(diff>1 and iCap+diff>255):return
		r = []
		eventlist = np.linspace(0,nevents,nevents+1,dtype=int)
		#Find the events that are not close to the trigger.
		for e in range(nevents):
			# Note: this cut will not 100% work if left bound of trigger octant is less than cap6
			if (iCap-trigger_pos[e]) >= -6 and (iCap-trigger_pos[e]) <= 40:
				continue
			elif (trigger_pos[e]+40) >= 256 and iCap <= (trigger_pos[e]+40)%256:
				continue
			# if (trigger_pos[e]-cap1)%256 <= 32 or (trigger_pos[e]-cap1)%256 >= 224:
			# 	continue
			# if  abs(trigger_pos[e]-cap1)<15 or (trigger_pos[e]-15<0 and 256-cap1+trigger_pos[e]<15) or (cap1-15<0 and 256-trigger_pos[e]+cap1<15):
			# 	continue
			else:
				r.append(e)
		x_tmp = (ydata2[r,channel, cap2] + ydata2[r,channel, cap1])
		y_tmp = (ydata2[r,channel, cap1] - ydata2[r,channel, cap2])
		eventlist = eventlist[r]
			
				
		# x_tmp = (ydata2[:,channel, cap2] + ydata2[:,channel, cap1])
		# y_tmp = (ydata2[:,channel, cap1] - ydata2[:,channel, cap2])
		indarray = np.argsort(x_tmp)
		xpeak_plus = np.median(x_tmp[indarray[-nevents//10:]])
		x_zero = np.median(x_tmp)

		indarray = np.argsort(y_tmp)
		ypeak_plus = np.median(y_tmp[indarray[-nevents//10:]])
		y_zero = np.median(y_tmp)

		#Remove outliers
		r0 = [r0 for r0 in range(len(y_tmp)) if ((x_tmp[r0]-x_zero)/xpeak_plus)**2+((y_tmp[r0]-y_zero)/ypeak_plus)**2<1.5]
		y = np.array(y_tmp[r0])
		x = np.array(x_tmp[r0])
		eventlist = eventlist[r0]

		fig, ax = plt.subplots()
		ax.set_title("Channel %d, cap %d vs cap %d, %d events"%(channel, cap1, cap2, len(x)))
		ax.set_xlim(1.1*np.min(x), 1.1*np.max(x))
		ax.set_ylim(1.1*np.min(y), 1.1*np.max(y))
		graph, = ax.plot([],[],'.')

		def animate(i):
			graph.set_data(x[:i+1],y[:i+1])
			return graph
		
		ani = FuncAnimation(fig, animate, frames=400, interval=100)
		ani.save("plots/ellipse_plot.gif", writer="imagemagick", fps=30)
		plt.show()

		return	
	
	def create_timebase_weighted(self):

		if not self.measurement_config["timebase"]["input"]:
			prefix = self.measurement_config["timebase"]["prefix"]
			files = glob.glob(prefix + "*")
			filedict = {}
			for file in files:
				multiplicity = file.count('ch')
				if multiplicity == 1:
					loc = file.index('ch') + 2
					filedict[file[loc:loc+2].strip('_')] = file
				else:
					loc = file.index('-ch') + 3
					filedict[file[loc:loc+2].strip('_')] = file
			index_maxes = np.sort([int(key) for key in filedict.keys()])

		timebase = np.zeros((30, 256), dtype=np.float64)
		for channel in self.measurement_config["timebase"]["channels"]:
			print(f'Starting on channel #{channel}...')
			if self.measurement_config["timebase"]["input"]:
				file_path = self.measurement_config["timebase"]["input"]
			else:
				file_path = filedict[str(index_maxes[channel <= index_maxes][0])]
			times320, _, sineData = Util.getDataRaw([file_path])
			trigger_pos = (((times320+2+2)%8)*32-16)%256
			trigger_pos += self.measurement_config["timebase"]["trigger_offset"]
			trigger_pos %= 256
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
				chTimeStdevMatrix = []
				coefs = []
				for iCap in range(256):
					x = []
					y =[]
					cap1 = iCap+256
					cap2 = iCap+256+diff
					if(diff>1 and iCap+diff>255):continue
					r = []
					eventlist = np.linspace(0,nevents,nevents+1,dtype=int)
					#Find the events that are not close to the trigger.
					for e in range(nevents):
						# Note: this cut will not 100% work if left bound of trigger octant is less than cap6
						if (iCap-trigger_pos[e]) >= -6 and (iCap-trigger_pos[e]) <= 40:
							continue
						elif (trigger_pos[e]+40) >= 256 and iCap <= (trigger_pos[e]+40)%256:
							continue
						# if (trigger_pos[e]-cap1)%256 <= 32 or (trigger_pos[e]-cap1)%256 >= 224:
						# 	continue
						# if  abs(trigger_pos[e]-cap1)<15 or (trigger_pos[e]-15<0 and 256-cap1+trigger_pos[e]<15) or (cap1-15<0 and 256-trigger_pos[e]+cap1<15):
						# 	continue
						else:
							r.append(e)
					x_tmp = (ydata2[r,channel, cap2] + ydata2[r,channel, cap1])
					y_tmp = (ydata2[r,channel, cap1] - ydata2[r,channel, cap2])
					eventlist = eventlist[r]
					
                        
					# x_tmp = (ydata2[:,channel, cap2] + ydata2[:,channel, cap1])
					# y_tmp = (ydata2[:,channel, cap1] - ydata2[:,channel, cap2])
					indarray = np.argsort(x_tmp)
					xpeak_plus = np.median(x_tmp[indarray[-nevents//10:]])
					x_zero = np.median(x_tmp)

					indarray = np.argsort(y_tmp)
					ypeak_plus = np.median(y_tmp[indarray[-nevents//10:]])
					y_zero = np.median(y_tmp)

					#Remove outliers
					r0 = [r0 for r0 in range(len(y_tmp)) if ((x_tmp[r0]-x_zero)/xpeak_plus)**2+((y_tmp[r0]-y_zero)/ypeak_plus)**2<1.5]
					y = y_tmp[r0]
					x = x_tmp[r0]
					eventlist = eventlist[r0]

					# if iCap == 30 and diff==1:
					# 	odd_events = eventlist[(x > -0.1) & (x < 0.1) & (y > -0.01) & (y < 0.01)]
					# 	print(odd_events)
					# 	print(len(odd_events))

					# Formulate and solve the least squares problem ||Ax - b ||^2
					A = np.column_stack([x**2, x * y, y**2, x, y])
					b = np.ones_like(x)
					lstsq = np.linalg.lstsq(A, b, rcond=None)
					fit = lstsq[0].squeeze()

					
					res = np.sqrt(lstsq[1] / (np.size(b)-2))# res is unitless because we are fitting the ellipse to 1, a unitless quantity.
					coefs.append(fit)
					#print(fit)
					try:
						if(np.shape(res)!=(1,)):
							raise ValueError("")
						a = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]+math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
						#print("a = %f"%a)
						b = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]-math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
						#print("b = %f"%b)

						dtij = math.atan(b/a)/(math.pi*true_freq) # This is roughly sqrt(fit[0]/fit[2])/(math.pi*true_freq)
						dfit0 = res * np.sqrt(np.linalg.inv(np.transpose(A)@A)[0,0])
						dfit2 = res * np.sqrt(np.linalg.inv(np.transpose(A)@A)[2,2])
						rough_uncertainty = 0.5*np.sqrt((dfit0/fit[0])**2 + (dfit2/fit[2])**2)
						#print("dtij = %f ps"%(dtij*1e12))
						chTimeOffsetMatrix.append(dtij)
						# The assumption here is that the ellipse is not rotated.
						chTimeStdevMatrix.append(rough_uncertainty*dtij)
						if(VERBOSE and diff == 1):
							plt.title("Channel %d, cap %d vs cap %d, t0 %d[ps], %d events"%(channel, iCap, iCap+1, dtij*1e12, len(x)))
							plt.scatter(x, y)
							# Plot the least squares ellipse
							x_coord = np.linspace(1.05*x.min(),1.05*x.max(),300)
							y_coord = np.linspace(1.05*y.min(),1.05*y.max(),300)
							X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
							Z_coord = fit[0] * (X_coord ** 2) + fit[1] * X_coord* Y_coord + fit[2] * Y_coord**2+ fit[3] * X_coord+ fit[4] * Y_coord
							plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
							plt.savefig("plots/channel%d_cap%d.png"%(channel, iCap))
							plt.clf()
					except ValueError:
						if(VERBOSE):
							print("Error in timebase calculation. Using default value. %d samples apart, %dth sample."%(diff, iCap)) 
						chTimeOffsetMatrix.append(100.0e-12*diff)
						chTimeStdevMatrix.append(np.array([30.0*diff]))
				vsize = 256-diff
				if(diff==1):
					vsize = 256

				arr = np.zeros((vsize,256))
				for i in range(vsize):
					for j in range(256):#Do not include wraparound terms for diff>1
						if(j-i>=0 and j-i<diff):
							arr[i,j] = 1
						if(i>=256 and j==256):
							arr[i,j] = 1
				a_matrix.append(arr)
				y_matrix.append(np.array(chTimeOffsetMatrix))
				w_matrix.append(np.array(chTimeStdevMatrix))
			#Normalization so that the sum of timebase is 25ns. This is done by adding a data point (with very small variation and t_0+...+t_255=25e-9.)
			a_matrix.append(np.ones((1,256)))
			y_matrix.append(np.array([25e-9]))
			w_matrix.append(np.array([np.array([np.min(np.concatenate(w_matrix, axis=0))/100])]))#Have a really small number
			timebase[channel] = np.linalg.lstsq(np.divide(np.concatenate(a_matrix, axis=0),np.concatenate(w_matrix, axis=0)), np.divide(np.concatenate(y_matrix, axis=None),np.concatenate(w_matrix, axis=0).squeeze()), rcond=None)[0].squeeze()
			if(VERBOSE):
				print(timebase[channel])
				print(np.sum(timebase[channel]))
				plt.title("Timebase Weighted")
				plt.xlabel("Sample Number")
				plt.ylabel("Timebase [s]")
				if(ERRORBAR):
					plt.errorbar(range(256), np.concatenate(y_matrix, axis=None)[0:256], np.concatenate(w_matrix, axis=0).squeeze()[0:256] * 1e-13, ecolor="black")
				plt.step(range(256), timebase[channel])
				plt.show()
		if(SAVE):
			for channel in self.measurement_config["timebase"]["channels"]:
				self.time_df[channel] = timebase[channel]#Keep the address to the array the same so that TTree can read it.
			self.save()
		return sineData

	def find_voltage_uncertainty(self, display=False):

		stddevs = np.zeros((30,256), dtype=np.float64)

		def gaussfit(x, N, sigma, mu):
			return (N/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)

		_, _, pedestalData = Util.getDataRaw([self.measurement_config["timebase"]["pedestalData"]])
		channels = self.measurement_config["timebase"]["channels"]
		pedestalData = self.linearize_voltage(pedestalData)
		pedestalData = pedestalData[:,channels,:]
		for iCh, chan in enumerate(channels):
			chData = pedestalData[:,iCh,:]
			for cap in range(256):
				if WRAPAROUND_ONLY and (cap != 0 and cap != 255):
					continue
				capData = chData[:,cap]
				histvals, binedges = np.histogram(capData, bins=100)
				bincenters = binedges[:-1] + 0.5*np.diff(binedges)[0]
				
				mu0 = np.mean(capData)
				sigma0 = 0.1*(np.max(capData)-mu0)
				N0 = 0.001*np.max(histvals)
				p0 = [N0, sigma0, mu0]

				popt, pcov = scipy.optimize.curve_fit(gaussfit, bincenters, histvals, p0=p0)

				stddevs[chan, cap] = np.std(capData)
				if display:
					fig, ax = plt.subplots()
					ax.hist(capData, bins=100)
					# domain = np.linspace(bincenters[0], bincenters[-1], 300)
					# ax.plot(domain, gaussfit(domain, *p0), color='green')
					# ax.plot(domain, gaussfit(domain, *popt), color='red')
					# ax.set_xlabel('Capacitor voltage (V)')
					# ax.set_title(f'Ch #{chan}, cap #{cap}')
					# ax.xaxis.set_ticks_position('both')
					# ax.yaxis.set_ticks_position('both')
					# plt.minorticks_on()
					plt.show()
				

		return stddevs

	def plot_ellipses_histeresis(self):
		timebase = np.zeros((30, 256), dtype=np.float64)

		if ERRORBAR:
			stddevs = self.find_voltage_uncertainty()

		subsetsize = self.measurement_config["timebase"]["subsetsize"]
		nevents = self.measurement_config["timebase"]["nevents"]
		binsize = self.measurement_config["timebase"]["binsize"]
		if not subsetsize:
			subsetsize = nevents
		
		if self.measurement_config["timebase"]["input"] != None:
			file_path = self.measurement_config["timebase"]["input"]
		else:	# not sure if this else statement is working like create_timebase_weighted()
			file_path = self.measurement_config["timebase"]["prefix"]+str(channel)+self.measurement_config["timebase"]["suffix"]

		channels = self.measurement_config["timebase"]["channels"]

		wraparounds = np.zeros((len(channels), nevents//binsize), dtype=np.float64)
		offsets_times = []

		# A subset is larger than a bin and is used to split up super large data so millions of events are not allocated to memory all at once
		subsetstart = 0
		while subsetstart < nevents:

			times320, _, sineData = Util.getDataRaw([file_path], startevent=subsetstart, batchsize=subsetsize)
			subsetstart += subsetsize

			trigger_pos = (((times320+2+2)%8)*32-16)%256
			trigger_pos += self.measurement_config["timebase"]["trigger_offset"]
			trigger_pos %= 256
			sineData = self.linearize_voltage(sineData) - 1.2/4096*self.measurement_config["timebase"]["pedestal"]
			true_freq = self.measurement_config["timebase"]["true_freq"]#Frequency of the signal source used for timebase measurement.
			
			ydata = sineData
			ydata2 = np.concatenate((ydata, ydata, ydata), axis=2)

			for iChannel, channel in enumerate(channels):
				
				a_matrix = []
				y_matrix = []
				w_matrix = []
				for diff in [1]:
					
					for iCap in range(256):
						full_x, full_y = [], []
						if WRAPAROUND_ONLY and iCap != 255:
							continue
						chTimeOffsetBinMatrix = []
						chTimeStdevBinMatrix = []
						binErrs = []
						for bin in range(subsetsize//binsize):
							repetitions = 1
							if ERRORBAR:
								repetitions = 1001
								dtijRep = []
							
							for iRep in range(repetitions):

								x = []
								y =[]
								coefs = []
								cap1 = iCap+256
								cap2 = iCap+256+diff
								if(diff>1 and iCap+diff>255):continue

								r = []
								eventlist = np.linspace(subsetstart,subsetstart+subsetsize-1,subsetsize,dtype=int)
								#Find the events that are not close to the trigger.
								for e in range(bin*binsize,min(bin*binsize+binsize, subsetsize)):
									# Note: this cut will not 100% work if left bound of trigger octant is less than cap6
									if (iCap-trigger_pos[e]) >= -6 and (iCap-trigger_pos[e]) <= 40:
										continue
									elif (trigger_pos[e]+40) >= 256 and iCap <= (trigger_pos[e]+40)%256:
										continue
									# if (trigger_pos[e]-cap1)%256 <= 32 or (trigger_pos[e]-cap1)%256 >= 224:
									# 	continue
									# if  abs(trigger_pos[e]-cap1)<15 or (trigger_pos[e]-15<0 and 256-cap1+trigger_pos[e]<15) or (cap1-15<0 and 256-trigger_pos[e]+cap1<15):
									# 	continue
									else:
										r.append(e)
								eventlist = eventlist[r]

								Vi = ydata2[r, channel, cap1]
								Vj = ydata2[r, channel, cap2]
								if ERRORBAR and iRep < repetitions-1:
									Vi += stddevs[channel, cap1%256]*np.random.normal()
									Vj += stddevs[channel, cap2%256]*np.random.normal()
									
								x_tmp = (Vj + Vi)
								y_tmp = (Vi - Vj)
								indarray = np.argsort(x_tmp)
								xpeak_plus = np.median(x_tmp[indarray[-binsize//10:]])
								x_zero = np.median(x_tmp)

								indarray = np.argsort(y_tmp)
								ypeak_plus = np.median(y_tmp[indarray[-binsize//10:]])
								y_zero = np.median(y_tmp)
								r0 = [r0 for r0 in range(len(y_tmp)) if ((x_tmp[r0]-x_zero)/xpeak_plus)**2+((y_tmp[r0]-y_zero)/ypeak_plus)**2<1.5]
								y = y_tmp[r0]
								x = x_tmp[r0]
								
								# Formulate and solve the least squares problem ||Ax - b ||^2
								
								A = np.column_stack([x**2, x * y, y**2, x, y])
								b = np.ones_like(x)
								lstsq = np.linalg.lstsq(A, b, rcond=None)
								fit = lstsq[0].squeeze()
								res = np.sqrt(lstsq[1]/(np.size(b)-2))# res is unitless because we are fitting the ellipse to 1, a unitless quantity.
								coefs.append(fit)
								#print(fit)
								try:
									if(np.shape(res)!=(1,)):
										raise Exception("")
									a = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]+math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
									#print("a = %f"%a)
									b = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]-math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
									#print("b = %f"%b)

									dtij = math.atan(b/a)/(math.pi*true_freq)
									if iChannel == 0:
										offsets_times.append(times320[bin*binsize])
									#print("dtij = %f ps"%(dtij*1e12))
			
									# if dtij < 680e-12:
									# 	num_in_region = np.sum((x[y < -0.1935] > -0.12) & (x[y < -0.1935] < 0.05))
									# 	fig, ax = plt.subplots()
									# 	ax.set_title(f'{dtij*1e12} ps, {num_in_region}')
									# 	ax.scatter(x, y)
									# 	ax.set_xlabel("320MHz clock %d to %d"%(times320[bin*binsize], times320[(bin+1)*binsize-1]))
									# 	# Plot the least squares ellipse
									# 	x_coord = np.linspace(1.05*x.min(),1.05*x.max(),300)
									# 	y_coord = np.linspace(1.05*y.min(),1.05*y.max(),300)
									# 	X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
									# 	Z_coord = fit[0] * (X_coord ** 2) + fit[1] * X_coord* Y_coord + fit[2] * Y_coord**2+ fit[3] * X_coord+ fit[4] * Y_coord
									# 	ax.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
									# 	plt.show()
									
									if iRep < repetitions-1:
										dtijRep.append(dtij)
									else:
										if ERRORBAR:
											fig, ax = plt.subplots()
											dtijRep = np.array(dtijRep)
											ax.hist(dtijRep*1e12, bins=50)
											ax.set_xlabel('$\Delta_{ij}$ (ps)', fontdict=dict(size=14))
											ax.set_title(f'# repetitions: {repetitions-1}, $\sigma={round(np.std(dtijRep)*1e12, 2)}$ ps', fontdict=dict(size=15))
											plt.show()
											binErrs.append(np.std(dtijRep))

										chTimeOffsetBinMatrix.append(dtij)
										chTimeStdevBinMatrix.append(res*dtij)
										wraparounds[iChannel, (subsetstart-subsetsize)//binsize + bin] = dtij

										if(not WRAPAROUND_ONLY or iCap == 255):
											plt.clf()
											if not ERRORBAR:
												plt.title(f"Channel {channel}, cap {cap1} vs cap {cap2}, t0 {round(dtij*1e12, decimals=2)}[ps], {len(x)} events")
											else:
												plt.title(f"Channel {channel}, cap {cap1} vs cap {cap2}, t0 {round(dtij*1e12, binErrs[-1]*1e12)}[ps], {len(x)} events")
											plt.scatter(x, y)
											plt.xlabel("320MHz clock %d to %d"%(times320[bin*binsize], times320[(bin+1)*binsize-1]))
											# Plot the least squares ellipse
											x_coord = np.linspace(1.05*x.min(),1.05*x.max(),300)
											y_coord = np.linspace(1.05*y.min(),1.05*y.max(),300)
											X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
											Z_coord = fit[0] * (X_coord ** 2) + fit[1] * X_coord* Y_coord + fit[2] * Y_coord**2+ fit[3] * X_coord+ fit[4] * Y_coord
											plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
											plt.savefig("plots/channel%d_cap%d_bin%d.png"%(channel, iCap, (subsetstart//subsetsize-1)*(subsetsize//binsize)+bin))
									
									full_x.append(x)
									full_y.append(y)
					
								except Exception as err:
									# raise err
									if iRep == 0:
										chTimeOffsetBinMatrix.append(100.0e-12*diff)
										chTimeStdevBinMatrix.append(np.array([30.0*diff]))

		if WRAPAROUND_ONLY:
			for iChannel, channel in enumerate(channels):

				offsets = wraparounds[iChannel,:]
				print(offsets)
				offsets_times = np.array(offsets_times).flatten()
				plt.clf()
				plt.title("Time Offset Change Over Time")
				plt.xlabel("Time [s]")
				plt.ylabel("Time offset (ps)")
				if not ERRORBAR:
					plt.step(offsets_times, 1e12*offsets)
				# else:
				#	# plt.errorbar(np.array([times320[bin*binsize]/320e6 for bin in range(nevents//binsize)]).flatten(), np.array(chTimeOffsetBinMatrix).flatten(), np.array(chTimeStdevBinMatrix).flatten(), ecolor="black")
				# 	plt.errorbar(np.array([times320[bin*binsize]/320e6 for bin in range(nevents//binsize)]).flatten(), np.array(chTimeOffsetBinMatrix).flatten(), np.array(binErrs), ecolor="black")
				plt.savefig("plots/channel%d_timeOffsetFluctuation.png"%(channel))

				offsets = np.array(offsets).flatten()
				plt.clf()
				plt.title(f"Time Offset Change Histogram,\n$\mu={round(np.mean(offsets)*1e12, decimals=2)}$ ps, $\sigma={round(np.std(offsets)*1e12, decimals=2)}$ ps")
				plt.xlabel("Time offset (ps)")
				plt.ylabel("Number of event-groupings")
				plt.hist(offsets*1e12, bins=100)
				plt.savefig("plots/channel%d_timeOffsetFluctuation_HIST.png"%(channel))

					# plt.clf()
					# fig, ax = plt.subplots()
					# ax.scatter(np.concatenate(full_x), np.concatenate(full_y))
					# plt.show()
		return sineData

	#Generates a data file and saves a timebase calibration file.

	def simulate_timebase_weighted(self):
		timebase = self.time_df
		freq = 250e6
		sineData = np.zeros(shape=(1000, 30, 256))
		trigger = np.zeros(shape=(1000))
		for event in range(1000):
			phase = random.uniform(0, 2*np.pi)
			trig = random.randrange(256)
			sineData[event] = np.roll(np.sin(2*np.pi*freq*np.cumsum(np.roll(timebase, (255-trig)+1, axis= 1), axis=1)+phase), trig-255, axis = 1)
			trigger[event] = trig
		true_freq = self.measurement_config["timebase"]["true_freq"]#Frequency of the signal source used for timebase measurement.
		nevents = self.measurement_config["timebase"]["nevents"]
		print(sineData[200 ,0])
		ydata = sineData
		ydata2 = np.concatenate((ydata, ydata, ydata), axis=2)
		for channel in self.measurement_config["timebase"]["channels"]:
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
					r = []
					for e in range(0,nevents):
						if cap1 <= trigger[e]+256 <= cap2:
							continue
						else:
							r.append(e)
					x.append(ydata2[r,channel, cap2] + ydata2[r,channel, cap1])
					y.append(ydata2[r,channel, cap1] - ydata2[r,channel, cap2])
					# Formulate and solve the least squares problem ||Ax - b ||^2

					A = np.column_stack([x[iCap]**2, x[iCap] * y[iCap], y[iCap]**2, x[iCap], y[iCap]])
					b = np.ones_like(x[iCap])
					lstsq = np.linalg.lstsq(A, b, rcond=None)
					fit = lstsq[0].squeeze()
					res = lstsq[1] / (1/fit[0]+1/fit[2])# rough estimation of the size of the ellipse.
					coefs.append(fit)
					#print(fit)
					try:
						if(np.shape(res)!=(1,)):
							raise Exception("")
						a = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]+math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
						#print("a = %f"%a)
						b = -math.sqrt(2*(fit[0]*fit[4]**2+fit[2]*fit[3]**2-fit[1]*fit[3]*fit[4]-(fit[1]**2-4*fit[0]*fit[2]))*(fit[0]+fit[2]-math.sqrt((fit[0]-fit[2])**2+fit[1]**2)))/(fit[1]**2 - 4*fit[0]*fit[2])
						#print("b = %f"%b)

						dtij = math.atan(b/a)/(math.pi*true_freq)
						#print("dtij = %f ps"%(dtij*1e12))
						chTimeOffsetMatrix.append(dtij)
						if(diff==1):
							chTimeVarMatrix.append(res/5)#Compensation for wraparound.
						else:
							chTimeVarMatrix.append(res)
						if(VERBOSE and diff == 1 and 0<=iCap<=15):
							plt.title("Channel %d, cap %d vs cap %d, t0 %d[ps]"%(channel, cap1, cap2, dtij*1e12))
							plt.scatter(x[iCap], y[iCap])
							# Plot the least squares ellipse
							x_coord = np.linspace(1.05*x[iCap].min(),1.05*x[iCap].max(),300)
							y_coord = np.linspace(1.05*y[iCap].min(),1.05*y[iCap].max(),300)
							X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
							Z_coord = fit[0] * (X_coord ** 2) + fit[1] * X_coord* Y_coord + fit[2] * Y_coord**2+ fit[3] * X_coord+ fit[4] * Y_coord
							plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
							plt.savefig("plots/channel%d_cap%d.png"%(channel, iCap))
							plt.show()
					except:
						chTimeOffsetMatrix.append(100.0e-12*diff)
						chTimeVarMatrix.append(np.array([30.0*diff]))
				vsize = 256-diff
				if(diff==1):
					vsize = 256

				arr = np.zeros((vsize,256))
				for i in range(vsize):
					for j in range(256):#Do not include wraparound terms for diff>1
						if(j-i>=0 and j-i<diff):
							arr[i,j] = 1
						if(i>=256 and j==256):
							arr[i,j] = 1
				a_matrix.append(arr)
				y_matrix.append(np.array(chTimeOffsetMatrix))
				w_matrix.append(np.array(chTimeVarMatrix))
			#Normalization so that the sum of timebase is 25ns
			a_matrix.append(np.ones((1,256)))
			y_matrix.append(np.array([25e-9]))
			w_matrix.append(np.array([np.array([1])]))#Have a really small number
			timebase[channel] = np.linalg.lstsq(np.divide(np.concatenate(a_matrix, axis=0),np.concatenate(w_matrix, axis=0)), np.divide(np.concatenate(y_matrix, axis=None),np.concatenate(w_matrix, axis=0).squeeze()), rcond=None)[0].squeeze()
			if(VERBOSE):
				print(timebase[channel])
				print(np.sum(timebase[channel]))
				plt.title("Timebase Weighted")
				plt.xlabel("Sample Number")
				plt.ylabel("Timebase [s]")
				plt.errorbar(range(256), np.concatenate(y_matrix, axis=None)[0:256], np.concatenate(w_matrix, axis=0).squeeze()[0:256] * 1e-13, ecolor="black")
				plt.step(range(256), timebase[channel])
				plt.show()
		return sineData

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
	
	def plot_with_timebase(self):
		events = self.measurement_config["plot"]["event"]
		times320, times, rawData = Util.getDataRaw([self.measurement_config["plot"]["input"]])
		sineData = self.linearize_voltage(rawData) - 1.2/4096*self.measurement_config["plot"]["pedestal"]
		sineData = np.concatenate((sineData,sineData),axis=2)
		timebase = np.cumsum(np.roll(np.concatenate((self.time_df, self.time_df), axis = 1), 1, axis=1), axis = 1)
		if(VERBOSE):
			print("320MHz counter modulo 8 histogram for all events:")
			print(np.histogram(times320 % 8, bins = [0, 1,2,3,4,5,6,7])[0])
		for event in events:
			plt.title("Time offset and voltage calibration")
			plt.xlabel("time [s]")
			plt.ylabel("Voltage [V]")
			for channel in [0,1,2,3,4,5]:
				plt.plot(timebase[channel], sineData[event, channel, :], label=str(channel))
			plt.axvline(timebase[channel][(((times320[event]) % 8))*32-16])
			plt.legend(loc="lower right")
			plt.show()
			plt.title("Time offset and voltage calibration")
			plt.xlabel("time [s]")
			plt.ylabel("Voltage [V]")
			for channel in [6,7,8,9,10,11]:
				plt.plot(timebase[channel], sineData[event, channel, :], label=str(channel))
			plt.axvline(timebase[channel][(((times320[event]) % 8))*32-16])
			plt.legend(loc="lower right")
			plt.show()
			plt.title("Time offset and voltage calibration")
			plt.xlabel("time [s]")
			plt.ylabel("Voltage [V]")
			for channel in [12,13,14,15,16,17]:
				plt.plot(timebase[channel], sineData[event, channel, :], label=str(channel))
			plt.axvline(timebase[channel][(((times320[event]) % 8))*32-16])
			plt.legend(loc="lower right")
			plt.show()
			plt.title("Time offset and voltage calibration")
			plt.xlabel("time [s]")
			plt.ylabel("Voltage [V]")
			for channel in [18,19,20,21,22,23]:
				plt.plot(timebase[channel], sineData[event, channel, :], label=str(channel))
			plt.axvline(timebase[channel][(((times320[event]) % 8))*32-16])
			plt.legend(loc="lower right")
			plt.show()
			plt.title("Time offset and voltage calibration")
			plt.xlabel("time [s]")
			plt.ylabel("Voltage [V]")
			for channel in [24,25,26,27,28,29]:
				plt.plot(timebase[channel], sineData[event, channel, :], label=str(channel))
			plt.axvline(timebase[channel][(((times320[event]) % 8))*32-16])
			plt.legend(loc="lower right")
			plt.show()

	def plot(self, mode='r'):

		# Metadata
		events = self.measurement_config["plot"]["event"]
		nevents = self.measurement_config["plot"]["nevents"]
		trigger_thresh = self.measurement_config["plot"]["trigger_threshold"]

		# Reads in file and calibrates it based off mode (tasks in .yml config file)
		times320, times, rawData = Util.getDataRaw([self.measurement_config["plot"]["input"]], batchsize=nevents)
		if mode == 'r':
			data = np.array(rawData, dtype=np.float64)
			pedDataPath = self.measurement_config["plot"]["pedestalData"]
			if pedDataPath: 
				_, _, pedestalData = Util.getDataRaw([pedDataPath])
				data -= pedestalData.mean(0)
		if mode == 'p':
			data = self.linearize_voltage(rawData) - 1.2/4096*self.measurement_config["plot"]["pedestal"]
			if trigger_thresh: trigger_thresh *= 1.2/4096.

		# Beginning and end of trigger octant
		trigger_pos = (((times320+2+2)%8)*32-16)%256
		trigger_high = (((times320+2+2)%8)*32+24)%256

		if(VERBOSE):
			trigger_octants = (trigger_pos-16)/32
			print('Trigger octant histogram:')
			print(np.histogram(trigger_octants, bins=np.linspace(0,8,9,dtype=int))[0])
			print('')
			fig, ax = plt.subplots()
			ax.hist(trigger_pos, bins=np.linspace(0, 256, 257))
			ax.set_xlabel('Trigger position (sample)', fontdict=dict(size=14))
			ax.set_ylabel('Number of events per 1 sample bins', fontdict=dict(size=14))
			ax.xaxis.set_ticks_position('both')
			ax.yaxis.set_ticks_position('both')
			plt.minorticks_on()
			plt.show()

		for event in events:
			if VERBOSE:
				print(f"320MHz: {times320[event]}")
				print(f"1PPS: {(times[event] >> 32 ) & 0xffffffff}")
				print(f"250MHz: {times[event] & 0xffffffff}")
			
			for chip in range(5):
				channels = np.linspace(6*chip, 6*chip+5, 6, dtype=int)
				plt.title(f"Time offset and voltage calibration, Event {event}")
				plt.xlabel("time [s]")
				plt.ylabel("Voltage [V]")
				for channel in channels:
					plt.plot(np.linspace(0, 255,256), data[event, channel, :], label=str(channel))
				plt.axvline(trigger_pos[event], color='green')
				plt.axvline(trigger_high[event], color='red')
				if trigger_thresh:
					plt.axhline(trigger_thresh, color='black')
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
			
	def jin_linearize_voltage(self, sineData):
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
	
	def linearize_voltage(self, sineData, debug = False):
		
		vccs = [[None]*256 for i in range(30)]
		# Imports voltage calib data and normalizes
		with uproot.open(self.measurement_config["calibration_file"]) as calib_file:

			# Gets numpy array axes are: channel, cap, voltage increment, and ADC type
			voltage_counts = np.reshape(calib_file["config_tree"]["voltage_count_curves"].array(library="np"), (30,256,256,2))
			voltage_counts[:,:,:,0] = voltage_counts[:,:,:,0]*1.2/4096.

			# Filter the data and make it monotonically increasing
			# voltage_counts[:,:,:,1] = savgol_filter(voltage_counts[:,:,:,1], 41, 2, axis=2)
			# reorder = np.argsort(voltage_counts[:,:,:,0], axis=2)
			# voltage_counts = np.take_along_axis(voltage_counts, reorder[:,:,:,np.newaxis], axis=2)
			# warnings.simplefilter('error')
			
			for ch in range(0, 30):
				for cap in range(0, 256):

					vert_mask = np.append((np.diff(voltage_counts[ch,cap,:,1]) > 4), False)
					vert_mask = vert_mask & np.roll(vert_mask, 1)
					adc_data = voltage_counts[ch, cap, vert_mask, 1]
					volt_data = voltage_counts[ch, cap, vert_mask, 0]
					if(VCC_NONLINEARITY):
						tck = scipy.interpolate.splrep(adc_data, volt_data, s=0.00005, k=3)
						single_bspline = scipy.interpolate.BSpline(*tck, extrapolate=True)
						vccs[ch][cap] = single_bspline
					else:
						vccs[ch][cap] = scipy.interpolate.interp1d(adc_data, volt_data, fill_value='extrapolate')
					if(debug):
						fig, ax = plt.subplots()
						ax.set_title(f'Ch: {ch}, cap: {cap}')
						ax.scatter(voltage_counts[ch,cap,:,1], voltage_counts[ch,cap,:,0], marker='.', color='black')
						fig_domain = np.linspace(0, 4096, 1000)
						ax.plot(fig_domain, vccs[ch][cap](fig_domain), color='red')
						plt.show()
	
		linDat = np.zeros_like(sineData, float)
		
		for ch in range(0,30):
			for cap in range(0,256):
				linDat[:,ch,cap] = np.clip(vccs[ch][cap](sineData[:,ch,cap]), -2, 2)

		return linDat
		
	def histogram_2d(self):
		# Metadata
		events = self.measurement_config["plot"]["event"]
		nevents = self.measurement_config["plot"]["nevents"]
		trigger_thresh = self.measurement_config["plot"]["trigger_threshold"]

		# Reads in file and calibrates it based off mode (tasks in .yml config file)
		times320, times, rawData = Util.getDataRaw([self.measurement_config["plot"]["input"]], batchsize=nevents)
		
		channels = np.array([5,4,3,2,1,0,11,10,9,8,7,6,17,16,15,14,13,12,23,22,21,20,19,18,29,28,27,26,25,24])
		#channels = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
		for event in events:
			xdata = np.linspace(0, 255,256)*25/256
			ydata = np.linspace(0, 29, 30)
			zdata = self.linearize_voltage(rawData[event,channels,:]) - 1.2/4096*self.measurement_config["plot"]["pedestal"]
			plt.clf()
			fig, ax1 = plt.subplots(1, 1)
			norm = matplotlib.colors.CenteredNorm()
			#norm = matplotlib.colors.SymLogNorm(10)
			cm = ax1.pcolormesh(xdata, ydata, zdata, norm=norm, cmap='bwr')
			fig.colorbar(cm, ax = ax1)
			ax1.set_xlabel("Time sample (ns)")
			ax1.set_ylabel("Channel")
			fig.tight_layout()
			plt.show()

if __name__ == "__main__":
	try:
		a = sys.argv[1]
	except(IndexError):
		a = None
	ut = Util(a)
	if ut.measurement_config["old_format"] == 'T':
		OLD_FORMAT = True
	if ut.measurement_config["debug"] == 'T':
		DEBUG = True
	if ut.measurement_config["verbose"] == 'T':
		VERBOSE = True
	if ut.measurement_config["vcc_nonlinear"] == 'T':
		VCC_NONLINEARITY = True
	if 'R' in ut.measurement_config["tasks"]:
		WRAPAROUND_ONLY = True
	if 'E' in ut.measurement_config["tasks"]:
		ERRORBAR = True
	if ut.measurement_config["save"] == 'T':
		SAVE = True
	if 'r' in ut.measurement_config["tasks"]:
		ut.plot()
	if 'p' in ut.measurement_config["tasks"]:
		ut.plot(mode='p')
	if 'P' in ut.measurement_config["tasks"]:
		ut.plot_with_timebase()
	if '2' in ut.measurement_config["tasks"]:
		ut.histogram_2d()
	if 'e' in ut.measurement_config["tasks"]:
		ut.plot_ellipse()
	if 'U' in ut.measurement_config["tasks"]:
		ut.find_voltage_uncertainty(display=True)
	if 'v' in ut.measurement_config["tasks"]:
		ut.create_voltage_curve()
	if 'V' in ut.measurement_config["tasks"]:
		ut.linearize_voltage(np.zeros((1,30,256)), debug=True)
	if 't' in ut.measurement_config["tasks"]:
		ut.create_timebase_simple()
	if 'w' in ut.measurement_config["tasks"]:
		ut.create_timebase_weighted()
	if 'h' in ut.measurement_config["tasks"]:
		ut.plot_ellipses_histeresis()
	if 'W' in ut.measurement_config["tasks"]:
		ut.simulate_timebase_weighted()
	if 'j' in ut.measurement_config["tasks"]:
		ut.phase_dist_between_channel()
	if 'n' in ut.measurement_config["tasks"]:
		ut.phase_dist_square_norm()
	



