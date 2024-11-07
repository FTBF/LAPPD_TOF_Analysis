import numpy as np
import glob
import sys
sys.path.append("../Acdc/")
import Acdc 


#A true script with no arguments, this is meant for pre-reducing
#the waveform data to front-load some of the computing time. May
#someday happen on a cluster, so this is just a hard coded argumented script. 

#Structure of this dict:
#key: acdc number
#value: dict with the following keys
#   obj: Acdc object itself
#   config: configuration file path 
#   infiles: list of input data files for that board

#this set of boards is used in the July 2024 proton data at test-beam
acdcs = {44:{"obj":None, "config":"../configs/acdc44.yml", "infiles": [], "pedfiles": []}, \
	43:{"obj":None, "config":"../configs/acdc43.yml", "infiles": [], "pedfiles": []}}

#Configure some data filepaths
#you'll likely want to keep paths as we may be committing/pushing
#multiple paths. Just use a comment
datadir = "../../data/20240701ProtonDataFTBF/"


#load the waveforms into the events attribute
for acdc_num, a in acdcs.items():
	a["obj"] = Acdc.Acdc(a["config"])

	#the configuration is now parsed, so we can find which station id
	#in order to parse the filetag at the end of the data filenames. 
	bnum = a["obj"].c["station_id"]

	#find all the data files for this board
	a["infiles"] = glob.glob(datadir + f"Raw_Proton*b{bnum}.txt")

	#find all the pedestal files for this board
	a["pedfiles"] = glob.glob(datadir + f"Raw_test*b{bnum}.txt")


	#go file by file and save pre-reduced data for each file.
	#The machinery in the ACDC class can instead handle all files,
	# it is so resource and RAM intensive that I instead opt to save
	#a pre-reduced output file for each input file, looping individually. 
	for f in a["infiles"]:
		if(f.replace(".txt", "_prereduced.p") in glob.glob(datadir + f"*prereduced.p")):
			print(f"Skipping {f}, already pre-reduced.")
			continue
		#the function can take a list of files, 
		#so I pass just the one we are working on
		a["obj"].load_raw_data_to_events([f]) 

		#loads root file containing linearity data, 
		#calibrates pedestals based on ADC/voltage and does
		#pedestal subtraction, and loads timebase calibration.
		#Within these pedfiles, it finds the closest file to our
		#events without using a future pedfile. 
		a["obj"].calibrate_waveforms(a["pedfiles"])
		a["obj"].write_events_to_file(f.replace(".txt", "_prereduced.p"))


