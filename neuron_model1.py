# Load ultrasound libraries from k-Wave
from kwave.data import Vector  # For handling 2D/3D coordinate vectors
from kwave.kgrid import kWaveGrid  # Sets up the spatial and temporal simulation grid
from kwave.kmedium import kWaveMedium  # Defines medium properties like sound speed
from kwave.ksensor import kSensor  # Represents ultrasound sensor locations
from kwave.ksource import kSource  # Defines source pressure distribution
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D  # Core 2D acoustic simulation engine
from kwave.options.simulation_execution_options import SimulationExecutionOptions  # GPU/CPU and other execution configs
from kwave.options.simulation_options import SimulationOptions  # Sets numerical & saving options
from kwave.utils.conversion import cart2grid  # Convert Cartesian coords to grid points
from kwave.utils.kwave_array import kWaveArray  # For managing array of ultrasound elements
from kwave.utils.colormap import get_color_map  # Optional color maps
from kwave.utils.mapgen import make_cart_circle, make_disc  # For making circular/transducer geometries
from kwave.utils.signals import reorder_binary_sensor_data  # Align sensor outputs
from numpy.ma.core import mvoid  # Masked array support
from pressure_physics_DTUSN import make_pressure  # Custom function to define pressure pattern

# Load NEURON simulation libraries
from neuron import h, gui  # NEURON simulator core
from neuron.units import V, ms, mV  # Units for voltage, time

# Load parameters and plotting functions
from model_parameters import model_parameters as mp
from neuron_ultrasound_model_plot import (
    ring_of_neurons, neuron_plot, soma_voltage_over_time,
    plot_spike_times, plot_spike_times_with_synaptic_weights,
    plot_simulation_masks, plot_sensor_data_image, plot_sensor_trace
)

import numpy as npx  # Extra numpy import (maybe used later)

# -------------------------
# Define a single neuron with confocal-derived geometry and biophysics
class Cell_Geometry:
    def __init__(self, gid, x, y, z, theta, model_parameters, cell_data):
        self.cell_data = cell_data
        self._gid = gid
        self.model_parameters = model_parameters
        self._setup_morphology()
        self.all = self.soma.wholetree()  # Get all sections

        # Load preprocessed external coordinates from confocal scan
        self.external_all = [
            cell_data["scaled_soma_coordinates_micrometers"],
            cell_data["scaled_dendrite_coordinates_micrometers"]
        ]

        self._setup_biophysics()

        # Set position and rotation
        self.x = self.y = self.z = 0
        h.define_shape()
        self._rotate_z(theta)
        self._set_position(x, y, z)

        # Record spikes and membrane voltage
        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)
        self._ncs = []  # Will store synaptic connections
        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)  # Record soma voltage

    def __repr__(self):
        return "{}[{}]".format(self.name, self._gid)

    def _set_position(self, x, y, z):
        # Shift internal and external morphology to desired position
        for sec_index, sec in enumerate(self.all):
            for i in range(sec.n3d()):
                sec.pt3dchange(
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z + sec.z3d(i),
                    sec.diam3d(i),
                )

        for sec_index, sec in enumerate(self.all):
            for i in range(len(self.external_all[sec_index])):
                x_ext = self.external_all[sec_index][i][0]
                y_ext = self.external_all[sec_index][i][1]
                z_ext = 0  # Assume flat morphology
                sec.pt3dadd(
                    x - self.x + x_ext,
                    y - self.y + y_ext,
                    z - self.z + z_ext,
                    sec.diam3d(i)
                )
        self.x, self.y, self.z = x, y, z

    def _rotate_z(self, theta):
        # Rotate neuron around z-axis by theta radians
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                x_prime = x * c - y * s
                y_prime = x * s + y * c
                sec.pt3dchange(i, x_prime, y_prime, sec.z3d(i), sec.diam3d(i))

# -------------------------
# Add morphology and biophysics to the cell
class NeuronCell_Morphology_Biophysics(Cell_Geometry):
    name = "NeuronCell_Morphology_Biophysics"

    def _setup_morphology(self):
        # Create basic soma-dendrite morphology
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = self.model_parameters["soma_length"]
        self.dend.L = self.model_parameters["dend_length"]
        self.dend.diam = self.model_parameters["dend_diameter"]

    def _setup_biophysics(self):
        # Assign passive properties to all sections
        for sec in self.all:
            self.Ra = self.model_parameters["axial_res"]
            sec.cm = self.model_parameters["membrane_cap"]

        # Insert Hodgkin-Huxley channels into soma
        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = self.model_parameters["Na_conductance"]
            seg.hh.gkbar = self.model_parameters["K_conductance"]
            seg.hh.gl = self.model_parameters["leak_conductance"]
            seg.hh.el = self.model_parameters["rev_potential"]

        # Insert passive and calcium channels in dendrite
        self.dend.insert("pas")
        self.dend.insert("ca")
        for seg in self.dend:
            seg.pas.g = self.model_parameters["passive_conductance"]
            seg.pas.e = self.model_parameters["leak_rev_potential"]
            # Optional: debug info about Ca dynamics
            if hasattr(seg, 'ca'):
                print(" ")

        # Create exponential synapse at middle of dendrite
        self.syn = h.ExpSyn(self.dend(0.5))
        self.syn.tau = 2 * ms

# -------------------------
# Define a ring network of connected neurons
class Interneuron_Group:
    """Creates a ring network of N neurons with circular geometry and recurrent synaptic connections"""
    def __init__(
        self,
        model_parameters,

        
        N=mp["number_neurons"],
        stim_w=mp["stimulus_weight"],
        stim_t=mp["stimulus_time"],
        stim_delay=mp["stimulus_delay"],
        syn_w=mp["synapse_weight"],
        syn_delay=mp["synapse_delay"],
        r=mp["radius"],
        cell_data=None
    ):
        
        # Store parameters
        self._syn_w = syn_w #mp["stimulus_weight"]
        self.model_parameters = model_parameters
        self._syn_delay = syn_delay #mp["synapse_delay"]

        # Build network
        self._create_cells(N, r, cell_data)
        self._connect_cells()

        # Add stimulus to first cell in the ring
        self._netstim = h.NetStim()
        self._netstim.number = 1
        self._netstim.start = stim_t #mp["stimulus_time"]
        self._nc = h.NetCon(self._netstim, self.cells[0].syn)
        self._nc.delay = stim_delay #mp["stimulus_delay"]
        self._nc.weight[0] = stim_w #mp["stimulus_weight"]

        self.set_pressure_point = []  # Custom tag (unused unless updated)
        self.cell_data = cell_data

    def _create_cells(self, N, r, cell_data):
        # Place N cells evenly around a circle of radius r
        self.cells = []
        for i in range(N):
            theta = i * 2 * h.PI / N
            self.cells.append(
                NeuronCell_Morphology_Biophysics(
                    i,
                    h.cos(theta) * r,
                    h.sin(theta) * r,
                    0,
                    theta,
                    self.model_parameters,
                    cell_data
                )
            )

    def _connect_cells(self):
        # Connect each cell to the next in a ring (N → 1)
        for source, target in zip(self.cells, self.cells[1:] + [self.cells[0]]):
            nc = h.NetCon(source.soma(0.5)._ref_v, target.syn, sec=source.soma)
            nc.weight[0] = self._syn_w
            nc.delay = self._syn_delay
            source._ncs.append(nc)

    def set_pressure_point(self, pressure_intensity):
        # Unused placeholder for tagging specific stimulation
        self.set_pressure_point = pressure_intensity

