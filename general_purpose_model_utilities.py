# renamed as general_purpose_model_utilities

from neuron import h, gui  # NEURON simulation interface
from neuron.units import V, ms, mV  # Units for voltage and time
from model_parameters import model_parameters as mp  # Load model constants
from neuron_model1 import NeuronCell_Morphology_Biophysics  # Custom neuron class

import json  # For reading coordinate data from JSON

# Load standard NEURON run system
h.load_file("stdrun.hoc")

# -----------------------------
# Load confocal cell morphology data from JSON
def load_extracted_cell_coordinates():
    with open("cell_data_scaled.json", "r") as f:
        return json.load(f)

# -----------------------------
# UNUSED: Template to create a circular ring of cells
def _create_cells(N, r, cell_data):
    cells = []
    for i in range(N):
        theta = i * 2 * h.PI / N
        cells.append(
            BallAndStick(i, h.cos(theta) * r, h.sin(theta) * r, 0, theta, self.model_parameters, cell_data)
        )

# UNUSED: Template to connect ring neurons with synapses
def _connect_cells(self):
    for source, target in zip(self.cells, self.cells[1:] + [self.cells[0]]):
        nc = h.NetCon(source.soma(0.5)._ref_v, target.syn, sec=source.soma)
        nc.weight[0] = self._syn_w
        nc.delay = self._syn_delay
        source._ncs.append(nc)

# UNUSED: Placeholder function for setting pressure point
def set_pressure_point(self, pressure_intensity):
    self.set_pressure_point = pressure_intensity

# -----------------------------
# Extracts x, y, z coords from the first dendrite of a cell
def extract_first_dendrite_points(cell):
    dendrite = cell.all[1]  # Assume 2nd section is dendrite
    x_coordinate = []
    y_coordinate = []
    z_coordinate = []

    # Loop through 3D points on the dendrite
    for i in range(dendrite.n3d()):
        x_coordinate.append(dendrite.x3d(i))
        y_coordinate.append(dendrite.y3d(i))
        z_coordinate.append(dendrite.z3d(i))

    # Return coordinates of the midpoint
    i = dendrite.n3d() // 2
    return [dendrite.x3d(i), dendrite.y3d(i), dendrite.z3d(i)]

# -----------------------------
# Run a full neuron + ultrasound simulation and return all outputs
def compute_action_potential(model_parameters, cell_data):
    source_points, sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid = ultrasound_simulation()
    ring = Ring(model_parameters, N=6, cell_data=cell_data)
    return ring, source_points, sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid

# -----------------------------
# Classify whether an action potential occurs (if V > 10 mV)
def classify_action_potential(model_parameters, cell_data):
    ring, source_points, sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid = compute_action_potential(model_parameters, cell_data)
    
    # Count time points where soma voltage > 10 mV
    peaks_array = np.sum(ring.cells[0].soma_v > 10)
    
    # Determine if spike occurred (any such time point in 100 ms run)
    high_in_first_100ms = peaks_array >= 1
    return high_in_first_100ms