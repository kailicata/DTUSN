
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
    #KL** source.p0 is where the inital pressure points are defined (Pa)
    source.p0 = make_pressure(N, Vector([N.x/4+ x_offset, N.y/4]),4)
    source_points = source.p0
    #source.p0[99:119, 59:199]=1
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


    return source_points , sensor_data, sensor_location, combined_sensor_data, logical_p0, pm1_mask, sensor, kgrid
