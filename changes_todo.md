# Aesthetic changes
- In the future during datataking, can the filenames have their file tags associated with the actual ACDC number, instead of "b0" and "b1"? Why create another "ID" unneccesarily?


# Functional changes
- Modularize a baseline subtraction function that looks at all samples before the pulse. 
- Rewrite function "determine_optimal_channel" to not have hard coding, use proper baseline subtraction
- Add argument to ACDC::process_files that handles pedestal files better. We will pass a list of pedestal files in parallel with a list of data files. This function should check for the closest-in-time pedestal file and, if it hasn't been calibrated by that file already, recalibrate using that file. 
- Timestamp management during data logging, and documenting how the timestamp counters work, reference, etc. Include data run start time in the metadata of a run.


# Organizational changes
- (Jin) Do some re-naming and/or create configurations such that the "Util.py" script/class, inside the scripts folder, is clearly a routine for creating calibration files. 




