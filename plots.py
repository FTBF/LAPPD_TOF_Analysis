import matplotlib.pyplot as plt
import Events
events = Events.Events()
if __name__ == "__main__":
    pass

def mass_spectrum_plot(events):
    numpy_array = events.ttree["Mass"].array(library="np") #get the mass array from the ttree
    pass
def angle_distribution_plot(events, station_num):
    pass
def angle_mass_plot(events, station_num):
    pass
def event_rate():
    pass

