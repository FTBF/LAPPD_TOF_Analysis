
# Data restructuring concept
in init():
self.derived_quantities = {
    "waveforms_optch" : data['waveforms_optch'],
    "waveforms_sin" : data['waveforms_sin'],
    "hpos" : data['hpos'],
    "vpos" : data['vpos'],
    "times_wr" : data['times_wr'],
    "eventphi" : data['eventphi'],
    "first_peak" : data['first_peak'],
    "phi" : data['phi'],
    "omega" : data['omega'],
    "delta_t" : data['delta_t'],
    "opt_chs" : data['opt_chs'],
    "chi2" : data['chi2'],
    "startcap" : data['startcap']
}

in load_npz():
self.derived_quantities["waveforms_optch"] = data['waveforms_optch']
self.derived_quantities["waveforms_sin"] = data['waveforms_sin']
self.derived_quantities["hpos"] = data['hpos']
self.derived_quantities["vpos"] = data['vpos']
self.derived_quantities["times_wr"] = data['times_wr']
self.derived_quantities["eventphi"] = data['eventphi']
self.derived_quantities["first_peak"] = data['first_peak']
self.derived_quantities["phi"] = data['phi']
self.derived_quantities["omega"] = data['omega']
self.derived_quantities["delta_t"] = data['delta_t']
self.derived_quantities["opt_chs"] = data['opt_chs']
self.derived_quantities["chi2"] = data['chi2']
self.derived_quantities["startcap"] = data['startcap']

in save_npz():
since np.savez only takes arrays, have 2 options
1) Keep using np.savez, and store each of the derived_quantities values individually. 
Downside is that it defeats the scalability of using a dictionary.
2) Find another way to save the values in the dictionary. 
Maybe pickle and store as binary?