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


