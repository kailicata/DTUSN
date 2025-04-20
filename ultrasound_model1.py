# Import ultrasound simulation libraries from k-Wave
from kwave.data import Vector  # Vector math for positions
from kwave.kgrid import kWaveGrid  # Simulation grid definition
from kwave.kmedium import kWaveMedium  # Acoustic medium properties
from kwave.ksensor import kSensor  # Defines the sensor setup
from kwave.ksource import kSource  # Defines the ultrasound source
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D  # 2D simulation solver
from kwave.options.simulation_execution_options import SimulationExecutionOptions  # Execution configs
from kwave.options.simulation_options import SimulationOptions  # Simulation options
from kwave.utils.conversion import cart2grid  # Converts Cartesian to grid
from kwave.utils.kwave_array import kWaveArray  # For managing ultrasound arrays
from kwave.utils.colormap import get_color_map  # Optional: for color mapping
from kwave.utils.mapgen import make_cart_circle, make_disc  # Generates geometric shapes
from kwave.utils.signals import reorder_binary_sensor_data  # Reorders k-Wave sensor data
from numpy.ma.core import mvoid  # For masked arrays
from pressure_physics_DTUSN import make_pressure  # Custom function for pressure input

# Import NEURON simulation libraries
from neuron import h, gui  # NEURON main API
from neuron.units import V, ms, mV  # NEURON units

# Import project-specific functions and parameters
from model_parameters import model_parameters as mp  # Loads all constants
from neuron_ultrasound_model_plot import (
    ring_of_neurons, neuron_plot, soma_voltage_over_time,
    plot_spike_times, plot_spike_times_with_synaptic_weights,
    plot_simulation_masks, plot_sensor_data_image, plot_sensor_trace
)

import numpy as npx  # For numerical operations

# Function to run the ultrasound simulation
def ultrasound_simulation():
    karray = kWaveArray()  # Initialize ultrasound array object

    # Load arc properties from config
    radius = mp["arc_radius"]
    diameter = mp["arc_diameter"]
    ring_radius = mp["ring_radius"]
    num_elemenst = mp["num_ultrasound_detectors"]

    print("started ")

    focus_pos = Vector([0,0])  # Focus point at center of grid

    # Position array elements in a circle
    element_pos = make_cart_circle(ring_radius, num_elemenst, focus_pos)

    # Manually shift sensors toward center of neuron ring
    element_pos[:,0] = Vector([12**-7,0])
    element_pos[:,1] = Vector([10**-7,0])

    # Offset all elements horizontally
    ultrasound_offset = [50*10**-6 , 50*10**-6]
    for i in range(num_elemenst):
        element_pos[:,i] = Vector([ultrasound_offset[0] + 10**-7 + i*10**-7 , ultrasound_offset[1]])

    # Add each arc element to array
    for idx in range(num_elemenst):
        karray.add_arc_element(element_pos[:, idx], radius, diameter, focus_pos)

    # Set up simulation grid
    N = Vector([mp["grid_x"], mp["grid_y"]])
    d = Vector([mp["displacement_x"], mp["displacement_y"]])
    kgrid = kWaveGrid(N, d)

    # Define acoustic medium
    medium = kWaveMedium(mp["sound_speed"])

    # Generate time array based on grid and medium
    kgrid.makeTime(medium.sound_speed)

    # Create pressure source
    source = kSource()
    x_offset = 20
    source.p0 = make_pressure(N, Vector([N.x/4 + x_offset, N.y/4]), 4)  # Pressure disc
    source_points = source.p0
    logical_p0 = source.p0.astype(bool)  # Convert to logical mask

    # Initialize sensor
    sensor = kSensor()
    sensor.mask = element_pos  # Set sensor positions

    # Simulation options
    simulation_options = SimulationOptions(
        save_to_disk=True,
        data_cast='single',
    )

    execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

    # First simulation run (reordering not used)
    output = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)

    # Reorder sensor data to match physical sensor layout
    _, _, reorder_index = cart2grid(kgrid, element_pos)

    # Redefine sensor mask using array object
    sensor.mask = karray.get_array_binary_mask(kgrid)
    sensor_location = sensor.mask  # Final sensor mask

    # Second simulation run with updated sensor mask
    output = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)

    # Raw sensor data: shape (140 surface points, 1207 time steps)
    sensor_data = output["p"].T

    # Combine surface data into one value per physical sensor
    combined_sensor_data = karray.combine_sensor_data(kgrid, sensor_data)

    # Create a mask for the PML boundary region for plotting
    pm1_size = simulation_options.pml_x_size
    pm1_mask = np.zeros((N.x, N.y), dtype=bool)
    pm1_mask[:pm1_size, :] = 1
    pm1_mask[:, :pm1_size] = 1
    pm1_mask[-pm1_size:, :] = 1
    pm1_mask[:, -pm1_size:] = 1

    return source_points, sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid