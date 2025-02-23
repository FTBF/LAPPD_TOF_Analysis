# Purpose
- Take raw data of LAPPD with ACDC Rev. C Frontend and turn them into prereduced files, using ACDC calibration files.
- Take the prereduced files and output the reduced files, per station.
- Interstation analysis: Compute the time of flight between multiple LAPPD stations, provided the synchronization channel in each station is taking the sine wave from the synchronized White Rabbit Zen Modules.

# TODO
- Check changes_todo.md for the current milestones.

# Examples
- To create prereduced data from raw ACDC output, check notebooks/Prereduce-Data.ipynb. Last tested with Python 3.11.10.
- To analyze the prereduced data, check notebooks/Example_Analysis.ipynb. Last tested with Python 3.11.10.

# Configuration files
- See configs/acdc43.yml for individual station analysis parameters. See analysis.yml for interstation analysis parameters.
- See reduced_quantities.yml and multistation_reduced_quantities.yml for output column names and their meanings.

# Calibration files
- Calibration files exist per ACDC board and accounts for voltage nonlinearity and the sampling time nonlinearity. They are in ROOT format and is not included in the git repository since the file size is big. Currently (Feb 2025) they are available on Evan's Google drive. If one has a physical access to an ACDC board and a sine wave generator, one can also create a calibration file using scripts/Calibration_And_Test.py utility. To use the utility, invoke the script from the terminal and give a configuration yaml file as a parameter. Examples of such configuration files are under scripts/Calibration_And_Test_Config_Yamls.  

# Core files
- See Util/Util.py for analysis algorithms.
- See Acdc/Acdc.py for the single station analysis flow. The main procedure for prereducing the raw data is calibrate_waveforms(). The main procedure for reducing the prereduced data is reduce_data().
- See Analysis/Analysis.py for the interstation analysis flow.

# Authors
Cameron Poe, Evan Angelico, Joe Pastika, Jinseo Park, Ahan Datta