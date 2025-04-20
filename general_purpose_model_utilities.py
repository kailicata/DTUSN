# renamed as general_purpose_model_utilities

from neuron import h,gui
from neuron.units import V, ms, mV 
from model_parameters import model_parameters as mp


import json


h.load_file("stdrun.hoc")

def load_extracted_cell_coordinates():
    #load dictionary of coordinates from json file
    with open("cell_data_scaled.json", "r") as f:
        return json.load(f)



def _create_cells(N, r, cell_data):
    cells = []
    for i in range(N):
        #offset_for_grid = (131.308, 227.432)
        theta = i * 2 *h.PI /N 
        cells.append(
                BallAndStick(i,  h.cos(theta) * r,  h.sin(theta) * r , 0, theta, self.model_parameters, cell_data)
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


def compute_action_potential(model_parameters, cell_data):
    source_points , sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid = ultrasound_simulation()
    ring = Ring(model_parameters,N=6, cell_data=cell_data)
    return ring, source_points , sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid 

def classify_action_potential(model_parameters, cell_data):
    ring, source_points , sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid  = compute_action_potential(model_parameters, cell_data)
    #print("type of ring cells soma v"+str(type(ring.cells[0].soma_v)))
    peaks_array = np.sum(ring.cells[0].soma_v > 10)
    high_in_first_100ms = peaks_array >= 1
    return high_in_first_100ms
