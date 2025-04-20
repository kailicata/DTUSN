from hashlib import file_digest
from neuron_model1 import Interneuron_Group, NeuronCell_Morphology_Biophysics, Cell_Geometry
from model_parameters import model_parameters as mp
from neuron_ultrasound_model_plot import neuron_plot
from general_purpose_model_utilities import load_extracted_cell_coordinates

cell_data = load_extracted_cell_coordinates()

ring = Interneuron_Group(mp,N=6, cell_data=cell_data)
neuron_plot(ring.cells[0])

