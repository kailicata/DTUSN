# Import neuron model classes
from neuron_model1 import Interneuron_Group, NeuronCell_Morphology_Biophysics, Cell_Geometry
# Import simulation parameters
from model_parameters import model_parameters as params # Import neuron visualization function
from neuron_ultrasound_model_plot import neuron_plot
# Import utility to load 3D confocal coordinates for soma/dendrite
from general_purpose_model_utilities import load_extracted_cell_coordinates



# Load preprocessed morphology data from confocal imaging
cell_data = load_extracted_cell_coordinates()
# Create a ring network of 6 neurons using the confocal morphology
def run_simulation(params):
    ring = Interneuron_Group(params["model"], N=params["N"], cell_data=params["cell_data"])
    return ring
# Plot the 3D morphology of the first neuron in the ring

ring = run_simulation(params)


neuron_plot(ring.cells[0])
