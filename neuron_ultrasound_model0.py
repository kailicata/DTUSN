#loading plotting libraries 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
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
#loading neuron libraries 
from neuron import h,gui
from neuron.units import ms, mV 


from model_parameters import model_parameters as mp

import numpy as np

h.load_file("stdrun.hoc")

def neuron_plot(ring):

    # Create an empty numpy array
    xs = np.array([])
    ys = np.array([])
    zs = np.array([])


    for cell in ring.cells:
        for sec in cell.all:
            for i in range(sec.n3d()):
                print(sec.x3d(i),sec.y3d(i),sec.z3d(i),)
                xs = np.append(xs, sec.x3d(i))
                ys = np.append(ys, sec.y3d(i))
                zs = np.append(zs, sec.z3d(i))


    
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")

    n = len(xs)

    ax.scatter(xs, ys, zs, marker='o')

    ax.set_xlabel('neuron cell len (micrometers)')
    ax.set_ylabel('neuron cell len (micrometers)')
    ax.set_zlabel('neuron cell len (micrometers)')

    plt.show()


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

    ultrasound_offset = [50*10**-6 , 50*10**-6]

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

    source = kSource()
    x_offset = 20
    #make a small disc in the top left of the domain
    source.p0 = make_disc(N, Vector([N.x/4+ x_offset, N.y/4]),4)
    source.p0[99:119, 59:199]=1
    logical_p0 = source.p0.astype(bool)
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


    #plot source, sensor, and pml masks

    #assign unique values to each mask
    sensor_val = sensor.mask * 1
    logical_p0_val = logical_p0 * 2
    pm1_mask_val = pm1_mask * 3


    #combine masks
    combined_mask = sensor_val + logical_p0_val + pm1_mask_val
    combined_mask = np.flipud(combined_mask)

    #define colormap
    colors = [
        (1,1,1), #white (background)
        (233/255, 131/255, 0/255), #orange (sensor)
        (253/255, 221/255, 92/255), #yellow(sources)
        (0.8, 0.8, 0.8), #light grey (PML mask)
    ]
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    fig, ax = plt.subplots(layout="tight", figsize=(10,4))
    ax.pcolormesh(combined_mask, cmap=cmap, shading="auto")
    plt.axis("image")

    #define labels for the color bar
    labels = {
        0: "None",
        1: "Sensor",
        2: "Inital pressure p0",
        3: "PML mask",
    }

    bounds = np.linspace(0, len(labels), len(labels)+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax2 = fig.add_axes([-100,-50,50,100])
    mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing="proportional", ticks=bounds, boundaries=bounds, format="%1i")

    ax.set_title("Simualtion of the Neural Ultrasonic Interaction (DTUSN)")
    ax.set_ylabel("Simulation Components[-]", size=12)

    #calculate the middle points for each segment of the colorbar
    mid_points = [(bounds[i] + 0*bounds[i+1])/2 for i in range(len(bounds)-1)]

    #set new tick positions adn labels
    ax2.set_yticks(mid_points)
    ax2.set_yticklabels(list(labels.values()))

    #plot recorded sensor data

 
    fig, [ax1, ax2] = plt.subplots(ncols=1, nrows=2)



    im2 = ax2.imshow(combined_sensor_data, aspect="auto", cmap=get_color_map(), interpolation="none")
    ax2.set_xlabel(r"Time [$\mu$s]")
    ax2.set_ylabel("Detector Number")
    ax2.set_title("Arc detector")
    fig.colorbar(im2, ax=ax2)

    plt.show()

    print("show plot")
    fig.subplots_adjust(hspace=0.5)

    #plot a trace from the recorded sensor data 
    fig = plt.figure()
    #plt.plot(kgrid.t_array.squeeze() * 1e6, sensor_data_point[0, :], label="Cartesian point detectors")
    plt.plot(kgrid.t_array.squeeze() *1e6, combined_sensor_data[0, :], label="Arc detecors")
    plt.xlabel(r"Time [$\mu$s]")
    plt.ylabel("pressure [pa]")
    plt.legend()

    plt.show()

#neuron simulation
class Cell:
    def __init__(self, gid, x,y, z, theta):
        self._gid = gid
        self._setup_morphology()
        self.all = self.soma.wholetree()
        self._setup_biophysics()
        self.x = self.y = self.z =0 
        h.define_shape()
        self._rotate_z(theta)
        self._set_position(x, y, z)

        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)

        self._ncs = []
        
        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)

    def __repr__(self):
        return"{}[{}]".format(self.name, self._gid)
    
    def _set_position(self, x, y, z):
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z + sec.z3d(i),
                    sec.diam3d(i),
                )
                #print( (i, x, y, z, sec.x3d(i), sec.y3d(i), sec.z3d(i)))
                print( (sec.x3d(i), sec.y3d(i)))
        self.x, self.y, self.z = x, y, z
    
    def _rotate_z(self, theta):
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                sec.pt3dchange(i, xprime, yprime, sec.z3d(i), sec.diam3d(i))
class BallAndStick(Cell):
    name = "BallAndStick"

    def _setup_morphology(self):
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = mp["soma_length"]
        self.dend.L = mp["dend_length"]
        self.dend.diam = mp["dend_diameter"]

    def _setup_biophysics(self):
        for sec in self.all:
            self.Ra = mp["axial_res"] #axial resistance in Ohm*cm
            sec.cm = mp["membrane_cap"] #membrane capactiance in micro farads/cm^s
        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = mp["Na_conductance"] #sodium conductance
            seg.hh.gkbar = mp["K_conductance"]#potassium conductance 
            seg.hh.gl = mp["leak_conductance"] #leak conductance
            seg.hh.el = mp["rev_potential"] #reversal potential in mV
        #passive current in the dendrite 
        self.dend.insert("pas")
        for seg in self.dend:
            seg.pas.g = mp["passive_conductance"] #passive conductance
            seg.pas.e = mp["leak_rev_potential"] #leak reversal potenatial mV

        self.syn = h.ExpSyn(self.dend(0.5))
        self.syn.tau = 2 * ms 
class Ring:
    """a network of *N* ball and stick cells where cell n makes 
    an excitory synapse onto cell n + 1 and the last, Nth cell in the network
    projects to the first cell"""
    def __init__(
            self, N= mp["number_neurons" ], stim_w=mp["stimulus_weight"], stim_t=mp["stimulus_time"], stim_delay=mp["stimulus_delay"], syn_w=mp["synapse_weight"],syn_delay=mp["synapse_delay" ],r=mp["radius"]
    ):
        """
        param N: numner of cells
        param stim_w: weight of the stimulus 
        param stim_t: time of the sitmulus(in ms)
        param stim_delay: delay of the stimulus (in ms)
        param syn_w:synaptic weight
        param syn_delay: delay of the synapse
        param r: radius of the network
        """
        self._syn_w = syn_w
        self._syn_delay = syn_delay
        self._create_cells(N, r)
        self._connect_cells()
        #add stimulus
        self._netstim = h.NetStim()
        self._netstim.number = 1
        self._netstim.start = stim_t
        self._nc = h.NetCon(self._netstim, self.cells[0].syn)
        self._nc.delay = stim_delay
        self._nc.weight[0] = stim_w

    def _create_cells(self, N, r):
        self.cells = []
        for i in range(N):
            #offset_for_grid = (131.308, 227.432)
            theta = i * 2 *h.PI /N 
            self.cells.append(
                BallAndStick(i,  h.cos(theta) * r,  h.sin(theta) * r , 0, theta)
            )

    def _connect_cells(self):
        for source, target in zip(self.cells, self.cells[1:] + [self.cells[0]]):
            nc = h.NetCon(source.soma(0.5)._ref_v, target.syn, sec=source.soma)
            nc.weight[0] = self._syn_w
            nc.delay = self._syn_delay
            source._ncs.append(nc)





if __name__ == "__main__":
    ultrasound_simulation()

ring = Ring(N=6)
neuron_plot(ring)


shape_window = h.PlotShape(True)
shape_window.show(0)

t = h.Vector().record(h._ref_t)
h.finitialize(-65 *mV)
h.continuerun(100)


"""


plt.plot(t, ring.cells[0].soma_v)
plt.show()

plt.figure()
for i, cell in enumerate(ring.cells):
    plt.vlines(list(cell.spike_times), i +0.5, i +1.5)
plt.show()


plt.figure()
for syn_w, color in [(0.01, "black"), (0.005, "red")]:
    ring = Ring(N=6, syn_w=syn_w)
    h.finitialize(-65* mV)
    h.continuerun(100 * ms)
    for i, cell in enumerate(ring.cells):
        plt.vlines(list(cell.spike_times), i +0.5, i +1.5, color = color)
plt.show()


"""



