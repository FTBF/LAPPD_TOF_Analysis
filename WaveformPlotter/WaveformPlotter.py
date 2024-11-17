import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../Acdc/")
import Acdc


#a class for organizing plotting utilities. Takes a list of ACDC objects
#with data that has been populated. 
class WaveformPlotter:
    def __init__(self, acdcs):
        self.acdcs = acdcs

    #returns the acdc indices in the list of acdcs
    #that have data loaded in. 
    def check_for_data(self):
        acdc_indices = []
        for i, a in enumerate(self.acdcs):
            if(len(a.events) > 0):
                acdc_indices.append(i)
        return acdc_indices
    

    def select_an_acdc(self, acdc_idx=None, acdc_id=None, station_id=None):
        selectors = {"idx": acdc_idx, "id": acdc_id, "station": station_id}
        if(all([s is None for s in selectors.values()])):
            print("You didn't select any ACDCs or stations, so I'm using the first one in my list.")
            acdc_idx = self.check_for_data()
            if(len(acdc_idx) == 0):
                print("No data loaded in any ACDCs.")
                return None
            
            acdc_idx = acdc_idx[0]

        else:
            acdc_idx = self.check_for_data()
            if(len(acdc_idx) == 0):
                print("No data loaded in any ACDCs, so I'm not plotting anything.")
                return 
            for s in selectors:
                if(selectors[s] is not None):
                    if(s == "idx" and (selectors[s] in acdc_idx)):
                        acdc_idx = selectors[s]
                    elif(s == "id"):
                        acdc_idx = [i for i in acdc_idx if self.acdcs[i].c["acdc_id"] == selectors[s]]
                        if(len(acdc_idx) == 0):
                            print(f"No data loaded in ACDC with ID {selectors[s]}, so I'm not plotting anything.")
                            return None
                        else:
                            acdc_idx = acdc_idx[0]
                    elif(s == "station"):
                        acdc_idx = [i for i in acdc_idx if self.acdcs[i].c["station_id"] == selectors[s]]
                        if(len(acdc_idx) == 0):
                            print(f"No data loaded in ACDC with ID {selectors[s]}, so I'm not plotting anything.")
                            return None
                        else:
                            acdc_idx = acdc_idx[0]
            
        return acdc_idx


    #description of arguments:
    #sep: mV separation between waveforms
    def plot_waveforms_separated(self, event_number=None, sep=50, acdc_idx=None, acdc_id=None, station_id=None, fig=None, ax=None):

        if(fig is None):
            fig, ax = plt.subplots()

        acdc_idx = self.select_an_acdc(acdc_idx, acdc_id, station_id)
        a = self.acdcs[acdc_idx]

        if(event_number is None):
            #use a random event
            rand_idx = np.random.choice(len(a.events))
            e = a.events[rand_idx]
        else:
            if(event_number >= len(a.events)):
                print(f"Event {event_number} is out of range for the number of events in this ACDC.")
                return fig, ax
            e = a.events[event_number]

        for ch, w in enumerate(e["waves"]):
            ts = np.cumsum(a.times[ch])
            ax.plot(ts, w + sep*ch, linewidth=1.5)
            ax.annotate(f"Strip {ch}", (ts[-1], w[-1] + sep*ch))
        
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("[mV]")

        return fig, ax
        