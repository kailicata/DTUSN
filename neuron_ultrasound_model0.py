#loading plotting libraries 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
#loading ultrasound libraries 
from kwave.data import Vector
from kwave.kgrid import kWaveGrid 
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor 
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.conversion import cart2grid
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.colormap import get_color_map
from kwave.utils.mapgen import make_cart_circle, make_disc
from kwave.utils.signals import reorder_binary_sensor_data
from numpy.ma.core import mvoid
from pressure_physics_DTUSN import make_pressure
#loading neuron libraries 
from neuron import h,gui
from neuron.units import V, ms, mV 


from model_parameters import model_parameters as mp
from neuron_ultrasound_model_plot import ring_of_neurons, neuron_plot, soma_voltage_over_time, plot_spike_times, plot_spike_times_with_synaptic_weights, plot_simulation_masks, plot_sensor_data_image, plot_sensor_trace


import numpy as npx

h.load_file("stdrun.hoc")

def load_extracted_cell_coordinates(scale_factor=0.01):
    with open("cell_data_scaled.json", "r") as f:
        confocal_microscopy_data = json.load(f)
    
    all_x = [coord[0] for coord in confocal_microscopy_data["scaled_soma_coordinates_micrometers"]] + \
    [coord[0] for coord in confocal_microscopy_data["scaled_dendrite_coordinates_micrometers"]]
    all_y = [coord[1] for coord in confocal_microscopy_data["scaled_soma_coordinates_micrometers"]] + \
    [coord[1] for coord in confocal_microscopy_data["scaled_dendrite_coordinates_micrometers"]]

    print("Max X (µm):", max(all_x))
    print("Max Y (µm):", max(all_y))

    for sec in confocal_microscopy_data["scaled_soma_coordinates_micrometers"]:
        sec[0] *= scale_factor
        sec[1] *= scale_factor

    for sec in confocal_microscopy_data["scaled_dendrite_coordinates_micrometers"]:
        sec[0] *= scale_factor
        sec[1] *= scale_factor
    
    return confocal_microscopy_data
    

def membrane_vibration(frequency, amplitude, time, duration):
    """
    Simulates membrane displacement over time due to ultrasound.

    Parameters:
        frequency (float): Frequency of ultrasound in Hz
        amplitude (float): Max displacement in micrometers
        time (numpy array): Time points in ms
        duration (float): Duration of ultrasound pulse in ms

    Returns:
        displacement (numpy array): Displacement values at each time point
    """
    omega = 2 * np.pi * frequency / 1000  # Convert Hz to ms⁻¹
    displacement = amplitude * np.sin(omega * time) * (time < duration)
    return displacement

def pressure_to_voltage(pressure_pa):
    """
    Convert pressure (in Pascals) to an estimated voltage shift (in mV).
    This is a placeholder linear function; refine with better model later.
    """
    alpha = 0.002  # sensitivity coefficient [mV/Pa], adjust if needed
    return alpha * pressure_pa

#ultrasound simulation

def ultrasound_simulation():
    #create empty array
    karray = kWaveArray()

    #define arc properties
    radius = mp["arc_radius"] #[m]
    diameter = mp["arc_diameter"]  #[m]
    ring_radius = mp["ring_radius"]  #[m]
    num_elemenst = mp["num_ultrasound_detectors"] 

    print("started ")
    #oreint all elements twoards center of grid
    focus_pos = Vector([0,0]) #[m]

    element_pos = make_cart_circle(ring_radius, num_elemenst, focus_pos) 

   # element_pos = Vector(Vector([0,0]))

    #KL** shift one sensor to center of stiumulation grid
    #KL** used 2 sensors because only array works. positioned at center of nuerons and 0,0 
    #p = Vector([12**-7,0])
    #t = element_pos[:,0]
    element_pos[:,0] = Vector([12**-7,0])
    element_pos[:,1] = Vector([10**-7,0])

    ultrasound_offset = [50*10**-6 , 50*10**-6]  # match segment positions

    for i in range(num_elemenst):
        element_pos[:,i] = Vector([ultrasound_offset[0] + 10**-7 + i*10**-7 ,ultrasound_offset[1] +0])



    #element_pos[:,1] = Vector([50*1**-6,30*1**-6])



    for idx in range(num_elemenst):
        karray.add_arc_element(element_pos[:, idx], radius, diameter, focus_pos)

    #grid properties
    N = Vector([mp["grid_x"], mp["grid_y"]])
    d = Vector([mp["displacement_x"], mp["displacement_y"]])
    kgrid = kWaveGrid(N,d)

    #medium properties
    medium = kWaveMedium(mp["sound_speed"])

    #time array
    #make time is from the medium sound speed and is in micro seconds
    kgrid.makeTime(medium.sound_speed)
    # Get time array in seconds
    source = kSource()
    x_offset = 20
    source.p_mask = make_pressure(N, Vector([N.x//4 + x_offset, N.y//4]), 4)
    logical_p0 = source.p_mask.astype(bool)

    # Get time array in seconds
    
    source_points = source.p_mask
    #source.p0[99:119, 59:199]=1
    #logical_p0 = source.p0.astype(bool)
    sensor = kSensor()
    sensor.mask = element_pos
    simulation_options = SimulationOptions(
        save_to_disk=True,
        data_cast='single',
    )

    execution_options = SimulationExecutionOptions(is_gpu_simulation=False)
    output = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)

    #reorder the sensor data returned by k-wave to match the order of the elemnsts in the array
    _, _, reorder_index = cart2grid(kgrid, element_pos)
    #sensor_data_point = reorder_binary_sensor_data(output["p"].T, reorder_index=reorder_index)

    sensor.mask = karray.get_array_binary_mask(kgrid)
    sensor_location = sensor.mask 

    output = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)
    #shape of sensor data is (140,1207) 
    #140 comes from sensor surface area points  
    #each surface area sensor point has 1207 pressure points 
    sensor_data = output["p"].T
    combined_sensor_data = karray.combine_sensor_data(kgrid, sensor_data)
    #KL**there are 3 sensors but 140 sesnor surface area points 
    #KL**for each sesnor they take into consideation each point of the surface areas 


    #visulaitzation

    #create pm1 mask
    pm1_size = simulation_options.pml_x_size #20
    pm1_mask = np.zeros((N.x, N.y), dtype=bool)
    pm1_mask[:pm1_size, :]=1
    pm1_mask[:, :pm1_size]=1
    pm1_mask[-pm1_size:, :]=1
    pm1_mask[:, -pm1_size:]=1

    print("Grid width (µm):", kgrid.Nx * kgrid.dx * 1e6)
    print("Grid height (µm):", kgrid.Ny * kgrid.dy * 1e6)


    return source_points , sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid


#neuron simulation
class Cell:
    def __init__(self, gid, x,y, z, theta, model_parameters, confocal_microscopy_data):
        self.cell_data = confocal_microscopy_data
        self._gid = gid
        self.model_parameters = model_parameters
        self._setup_morphology()

        self.all = self.soma.wholetree()

        self.external_all = [confocal_microscopy_data["scaled_soma_coordinates_micrometers"], confocal_microscopy_data["scaled_dendrite_coordinates_micrometers"]]

        self._setup_biophysics()

        self.x = self.y = self.z =0 
        h.define_shape()

        #rotate the external coordinates 
        self._rotate_external_coordinates(theta)
        #setting 
        self._set_position(x, y, z)
        
        
        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)
        self._ncs = []
        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)

    def __repr__(self):
        return"{}[{}]".format(self.name, self._gid)
    
    def _set_position(self, x, y, z):

        """
        #setting position of the cell memebrane points 
        for sec_index,sec in enumerate(self.all):
            #these are the orginal segment points from the simultor which we dont want 
            #KL we dont want them. in the future delete them 
            for i,p in enumerate(sec):
                p[0] = x - self.x + p[0]
                p[1] = y - self.y + p[1]
                p[0] = x - self.x + p[0]
        """

    
        #these are the new external segment points from the confocal image  
        #the json contaisn 3 griups of coordinates that came from the confocal image 
        for sec_index,sec in enumerate(self.all):
            for i in range(len(self.external_all[sec_index])):
                x_external = self.external_all[sec_index][i][0]
                y_external = self.external_all[sec_index][i][1]
                z_external = 0
                #dvec = h.Vector([0])
                # adding the external segment points 
                sec.pt3dadd(
                    #these are the new external segment points 
                    x - self.x + x_external,
                    y - self.y + y_external,
                    z - self.z + z_external,
                    sec.diam3d(i)
                )

                #print( (i, x, y, z, sec.x3d(i), sec.y3d(i), sec.z3d(i)))
                #print( (sec.x3d(i), sec.y3d(i)))
        #this is the origin of the entire cell 
        self.x, self.y, self.z = x, y, z 
       #print("self.x, self.y, self.z are " + str(self.x) + ", " + str(self.y) + ", " + str(self.z))
    
    def _rotate_external_coordinates(self, theta):
        for sec in self.external_all:
            for p in sec:
                x = p[0]
                y = p[1]
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                p[0] = xprime
                p[1] = yprime

                #sec.pt3dchange(p, xprime, yprime, sec.z3d(p), sec.diam3d(p))
                #print("sec.x3d(i), sec.y3d(i) are " + str(sec.x3d(i)) + ", " + str(sec.y3d(i)))

class BallAndStick(Cell):
    name = "BallAndStick"

    def _setup_morphology(self):
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = self.model_parameters["soma_length"]
        self.dend.L = self.model_parameters["dend_length"]
        self.dend.diam = self.model_parameters["dend_diameter"]

    def _setup_biophysics(self):
        for sec in self.all:
            self.Ra = self.model_parameters["axial_res"] #axial resistance in Ohm*cm
            sec.cm = self.model_parameters["membrane_cap"] #membrane capactiance in micro farads/cm^s
        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = self.model_parameters["Na_conductance"] #sodium conductance
            seg.hh.gkbar = self.model_parameters["K_conductance"]#potassium conductance 
            seg.hh.gl = self.model_parameters["leak_conductance"] #leak conductance
            seg.hh.el = self.model_parameters["rev_potential"] #reversal potential in mV
        #passive current in the dendrite 
        self.dend.insert("pas")
        self.dend.insert("ca")
        for seg in self.dend:
            seg.pas.g = self.model_parameters["passive_conductance"] #passive conductance
            seg.pas.e = self.model_parameters["leak_rev_potential"] #leak reversal potenatial mV
            """
            
            PARAMETER {
                gca = 0.0001 (S/cm2)  : Maximum conductance
                eca = 120 (mV)        : Reversal potential for Ca
            }
            """

            
            #**KL there is also eca and ica in seg and v is in section 
            if hasattr(seg, 'ca') == True:
                """
                print("this is the   Maximum conductance in calcium channe;" + str(seg.ca.gca))
                print("this is the  Reversal potential for Ca in calcium channel" + str(seg.ca.ica))
                """
                print(" ")
            """
            print("this is the leak reversal potential in the dendrite" + str(seg.pas.e))
            print("this is the passive conductance in the dendrite" + str(seg.pas.g))
            """


        self.syn = h.ExpSyn(self.dend(0.5))
        self.syn.tau = 2 * ms 

    def apply_mechanosensitive_modulation(self, vibration_amp):
        """
        Modifies K+ and Ca2+ channel conductances based on membrane vibration amplitude.

        Parameters:
            vibration_amp (float): Max vibration amplitude (e.g., in micrometers)
        """
        scaling_factor_k = 1 + 0.3 * vibration_amp  # e.g., 30% increase per micron stretch
        scaling_factor_ca = 1 + 0.2 * vibration_amp

        for seg in self.soma:
            seg.hh.gkbar *= scaling_factor_k  # stretch-sensitive K+ channel (TRAAK/TREK-1)
        
        for seg in self.dend:
            if hasattr(seg, 'ca'):
                seg.ca.gca *= scaling_factor_ca  # stretch-sensitive Ca²⁺ channel (PIEZO)

    def apply_local_pressure_modulation(self, segment_pressures):
        """
        Modulates each segment’s ion channels based on its local pressure.
        """
        idx = 0
        for sec in self.all:
            for seg in sec:
                if idx < len(segment_pressures):
                    pressure = segment_pressures[idx]
                    delta_v = pressure_to_voltage(pressure)

                    # Example: scale K+ and Ca2+ channels
                    if hasattr(seg, 'hh'):
                        seg.hh.gkbar *= 1 + 0.3 * delta_v  # K+ modulation
                    if hasattr(seg, 'ca'):
                        seg.ca.gca *= 1 + 0.2 * delta_v  # Ca2+ modulation

                    idx += 1




class Ring:
    """a network of *N* ball and stick cells where cell n makes 
    an excitory synapse onto cell n + 1 and the last, Nth cell in the network
    projects to the first cell"""
    def __init__(
            self,model_parameters, N=mp["number_neurons" ],stim_w=mp["stimulus_weight"], stim_t=mp["stimulus_time"], stim_delay=mp["stimulus_delay"], 
            syn_w=mp["synapse_weight"],syn_delay=mp["synapse_delay" ],r=mp["radius"],confocal_microscopy_data=None):
        """
        param N: numner of cells
        param stim_w: weight of the stimulus 
        param stim_t: time of the sitmulus(in ms)
        param stim_delay: delay of the stimulus (in ms)
        param syn_w:synaptic weight
        param syn_delay: delay of the synapse
        param r: radius of the network
        param model_parameters: dictionary of model parameters
        """ 
        self._syn_w = syn_w
        self.model_parameters = model_parameters
        self._syn_delay = syn_delay
        self._create_cells(N, r, confocal_microscopy_data)
        self._connect_cells()
        #add stimulus
        self._netstim = h.NetStim()
        self._netstim.number = 1
        self._netstim.start = stim_t
        self._nc = h.NetCon(self._netstim, self.cells[0].syn)
        self._nc.delay = stim_delay
        self._nc.weight[0] = stim_w
        self.set_pressure_point = []
        self.confocal_microscopy_data = confocal_microscopy_data


    def _create_cells(self, N, r, confocal_microscopy_data):
        self.cells = []
        for i in range(N):
            #offset_for_grid = (131.308, 227.432)
            theta = i * 2 *h.PI /N 
            self.cells.append(
                BallAndStick(i,  h.cos(theta) * r,  h.sin(theta) * r , 0, theta, self.model_parameters, confocal_microscopy_data)
            )

    def _connect_cells(self):
        for source, target in zip(self.cells, self.cells[1:] + [self.cells[0]]):
            nc = h.NetCon(source.soma(0.5)._ref_v, target.syn, sec=source.soma)
            nc.weight[0] = self._syn_w
            nc.delay = self._syn_delay
            source._ncs.append(nc)
    
    def set_pressure_point(self,pressure_intensity):
        self.set_pressure_point = pressure_intensity



def extract_first_dendrite_points(cell):
    dendrite = cell.all[1]
    x_coordinate = []
    y_coordinate = []
    z_coordinate = []
    sec = dendrite
    for i in range(sec.n3d()):
        x = sec.x3d(i)
        x_coordinate.append(x)
        y = sec.y3d(i)
        y_coordinate.append(y)
        z = sec.z3d(i)
        z_coordinate.append(z)
        """
        print(" ")
        print("neuron dendrite x " + str(i) + " coordinate: " + str(x))
        print(" ")
        print("neuron dendrite y " + str(i) + " coordinate: " + str(y))
        print(" ")
        print("neuron dendrite z " + str(i) + " coordinate: " + str(z))
        print(" ")
        """
    i = sec.n3d()//2 
    x, y, z = sec.x3d(i),sec.y3d(i),sec.z3d(i)
    #print(x,y,z)
    return [x, y, z]

def get_pressure_at_segment_locations(cell, pressure_grid, kgrid):
    """
    Maps ultrasound pressure from source.p0 (2D pressure field) to neuron segment positions.

    Parameters:
        cell: BallAndStick neuron
        pressure_grid: np.ndarray of shape [Nx, Ny]
        kgrid: kWaveGrid object

    Returns:
        segment_pressures: List of pressure values for each segment
    """
    segment_pressures = []

    dx_um = kgrid.dx * 1e6
    dy_um = kgrid.dy * 1e6
    Nx = kgrid.Nx
    Ny = kgrid.Ny

    for sec_coords in cell.external_all:
        for x_um, y_um in sec_coords:
            x_idx = int(x_um / dx_um)
            y_idx = int(y_um / dy_um)

            #print("x_um:", x_um, "dx_um:", dx_um, "x_idx:", x_idx, "Nx:", Nx)

            if 0 <= x_idx < Nx and 0 <= y_idx < Ny:
                pressure = pressure_grid[x_idx, y_idx]
            else:
                pressure = 0

            segment_pressures.append(pressure)

    return segment_pressures


def compute_action_potential(model_parameters, confocal_microscopy_data):
    source_points , sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid = ultrasound_simulation()
    ring = Ring(model_parameters,mp["number_neurons"], confocal_microscopy_data=confocal_microscopy_data)
    # Simulate 5 MHz ultrasound for 30 ms with 0.8 µm amplitude
    time = np.linspace(0, 50, 5000)  # 0 to 50 ms
    vibration = membrane_vibration(frequency=5e6,amplitude=0.8, time=time, duration=30)

    # Get the peak displacement
    vibration_amp = np.max(np.abs(vibration))

    # Apply it to the cell's gating
    for cell in ring.cells:
        cell.apply_mechanosensitive_modulation(vibration_amp)

    return ring, source_points , sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid,vibration_amp

def classify_action_potential(model_parameters, confocal_microscopy_data):
    ring, source_points , sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid,vibration_amp   = compute_action_potential(model_parameters, confocal_microscopy_data)
    #print("type of ring cells soma v"+str(type(ring.cells[0].soma_v)))
    """
    print(np.sum(ring.cells[0].soma_v > 1000))
    peaks_array = 0 
    high_in_first_100ms = peaks_array >= 1
    return high_in_first_100ms
    """
    soma_v_np = np.array(ring.cells[0].soma_v.to_python())
    spikes = np.sum(soma_v_np > 0)
    return spikes > 0


if __name__ == "__main__":
    confocal_microscopy_data = load_extracted_cell_coordinates()

    ring, source_points , sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid,vibration_amp  = compute_action_potential(mp, confocal_microscopy_data)
    p_first_dendrite = extract_first_dendrite_points(ring.cells[0])
    
    cell = ring.cells[0]
    

    pressure_frame = combined_sensor_data  # already 2D 
    segment_pressures = get_pressure_at_segment_locations(cell, pressure_frame, kgrid)

    

    print("Number of segments with assigned pressures:", len(segment_pressures))
    print("Example pressures:", segment_pressures[:5])

    cell.apply_local_pressure_modulation(segment_pressures)


    #PLOTS 

    plotting_on = True
    models_on = True 
    data_graphs_on = False 
    print_result = True
    

    if print_result:
        cell = ring.cells[0]
        print("Original gkbar:", cell.model_parameters["K_conductance"])
        print("Modulated gkbar:", cell.soma(0.5).hh.gkbar)
        print("Applied vibration amplitude:", vibration_amp)
        print(" ")
        ap_occurred = classify_action_potential(mp, confocal_microscopy_data)
        print("action potential occured = " + str(ap_occurred))

        


    if plotting_on:
        if models_on:
            #Neuron plot
            ring_of_neurons(ring)
            neuron_plot(ring.cells[0])
            #Neuron and ultrasound plot
            
            #neuron_and_ultrasound_plot(ring,source_points,sensor_data,sensor_location, p_first_dendrite)
            
        if data_graphs_on:

            # Plot segment pressures
            plt.figure(figsize=(10, 4))
            plt.plot(segment_pressures)
            plt.title("Ultrasound Pressure per Segment")
            plt.xlabel("Segment Index")
            plt.ylabel("Pressure (Pa)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            #Simulation masks
            plot_simulation_masks(sensor, logical_p0, pm1_mask)
            #Sensor data image
            plot_sensor_data_image(combined_sensor_data)
            #Sensor trace
            plot_sensor_trace(kgrid, combined_sensor_data)
            #Soma voltage over time
            soma_voltage_over_time(ring,mp,h, mV,ms)
            """
            plot_spike_times(ring,mp)
            plot_spike_times_with_synaptic_weights(ring, mp, h, mV, ms, Ring, cell_data)
            """

            
    print(" ")
    print("------------------------------------------")
    print(" ")
    print("Simulation Complete!")





"""

new_list = []

for i in sensor_data[0]:
    new_list.append(i)

print(new_list)


"""



