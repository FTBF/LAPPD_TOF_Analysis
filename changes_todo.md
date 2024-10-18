# Aesthetic changes
- In the future during datataking, can the filenames have their file tags associated with the actual ACDC number, instead of "b0" and "b1"? Why create another "ID" unneccesarily?


# Functional changes
- (Jin) Modularize a baseline subtraction function (line 672) that looks at all samples before the pulse. 
- Why is the data so noisy? Look at pedistal subtraction. Line 194, look at vccs in calibrate_board()
- Rewrite function "determine_optimal_channel" to not have hard coding, use proper baseline subtraction
- Add argument to ACDC::process_files that handles pedestal files better. We will pass a list of pedestal files in parallel with a list of data files. This function should check for the closest-in-time pedestal file and, if it hasn't been calibrated by that file already, recalibrate using that file. 
- Timestamp management during data logging, and documenting how the timestamp counters work, reference, etc. Include data run start time in the metadata of a run.
- Look at wraparound and reflect indices