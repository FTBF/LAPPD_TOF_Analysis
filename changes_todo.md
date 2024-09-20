# Aesthetic changes
- In the ACDC class, the config_data is unpacked in the initializer. But all of the variables are just the same names given by the keys in the config dict. Instead, just create a `self.c = config_data` (the yaml parsed dict) and everywhere that you need, say the acdc_id, just put self.c["acdc_id"]. That way, whenever you add more to this config, you don't need to creat a bunch of new attributes, you can just use your variable wherever (and search for it more easily as well). 
- In the future during datataking, can the filenames have their file tags associated with the actual ACDC number, instead of "b0" and "b1"? Why create another "ID" unneccesarily?
- (Ahan) Separate "utility functions" where they are not married to class structure. Replace calls to that function with the link to "util.py" 


# Functional changes
- Add argument to ACDC::process_files that handles pedestal files better. We will pass a list of pedestal files in parallel with a list of data files. This function should check for the closest-in-time pedestal file and, if it hasn't been calibrated by that file already, recalibrate using that file. 
- (Evan) Change data outputs into dictionaries so that one doesn't have to manually name every attribute. Then adjust (everything) the ACDC::save_npz functionality - see the comment at that function. For example `output_dict = {"first_peak":[], "reduced_quantity2":[]}`



# Organizational changes
- (Jin) Do some re-naming and/or create configurations such that the "Util.py" script/class, inside the scripts folder, is clearly a routine for creating calibration files. 




