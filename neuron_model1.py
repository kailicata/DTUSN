
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


#neuron simulation
class Cell_Geometry:
    def __init__(self, gid, x,y, z, theta, model_parameters, cell_data):
        self.cell_data = cell_data
        self._gid = gid
        self.model_parameters = model_parameters
        self._setup_morphology()

        self.all = self.soma.wholetree()

        self.external_all = [cell_data["scaled_soma_coordinates_micrometers"], cell_data["scaled_dendrite_coordinates_micrometers"]]

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
        #setting position of the cell memebrane points 
        for sec_index,sec in enumerate(self.all):
            #these are the orginal segment points from the simultor which we dont want 
            #KL we dont want them. in the future delete them 
            for i in range(sec.n3d()):
                sec.pt3dchange(
                    #these are the orginal segment points 
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z + sec.z3d(i),
                    sec.diam3d(i),
                )
        #these are the new external segment points from the confocal image  
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
                #print("sec.x3d(i), sec.y3d(i) are " + str(sec.x3d(i)) + ", " + str(sec.y3d(i)))

class NeuronCell_Morphology_Biophysics(Cell_Geometry):
    name = "NeuronCell_Morphology_Biophysics"

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








class Interneuron_Group:
    """a network of *N* ball and stick cells where cell n makes 
    an excitory synapse onto cell n + 1 and the last, Nth cell in the network
    projects to the first cell"""
    def __init__(
            self,model_parameters, N= mp["number_neurons" ], stim_w=mp["stimulus_weight"], stim_t=mp["stimulus_time"], stim_delay=mp["stimulus_delay"], 
            syn_w=mp["synapse_weight"],syn_delay=mp["synapse_delay" ],r=mp["radius"], cell_data=None):
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
        self._create_cells(N, r, cell_data)
        self._connect_cells()
        #add stimulus
        self._netstim = h.NetStim()
        self._netstim.number = 1
        self._netstim.start = stim_t
        self._nc = h.NetCon(self._netstim, self.cells[0].syn)
        self._nc.delay = stim_delay
        self._nc.weight[0] = stim_w
        self.set_pressure_point = []
        self.cell_data = cell_data


    def _create_cells(self, N, r, cell_data):
        self.cells = []
        for i in range(N):
            #offset_for_grid = (131.308, 227.432)
            theta = i * 2 *h.PI /N 
            self.cells.append(
                NeuronCell_Morphology_Biophysics(i,  h.cos(theta) * r,  h.sin(theta) * r , 0, theta, self.model_parameters, cell_data)
            )

    def _connect_cells(self):
        for source, target in zip(self.cells, self.cells[1:] + [self.cells[0]]):
            nc = h.NetCon(source.soma(0.5)._ref_v, target.syn, sec=source.soma)
            nc.weight[0] = self._syn_w
            nc.delay = self._syn_delay
            source._ncs.append(nc)
    
    def set_pressure_point(self,pressure_intensity):
        self.set_pressure_point = pressure_intensity



