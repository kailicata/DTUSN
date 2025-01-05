import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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


def main():
    #create empty array
    karray = kWaveArray()

    #define arc properties
    radius = 5e-6 #[m]
    diameter = 10e-6#[m]
    ring_radius = 50e-3 #[m]
    num_elemenst = 20

    print("started ")
    #oreint all elements twoards center of grid
    focus_pos = Vector([0,0]) #[m]

    element_pos = make_cart_circle(ring_radius, num_elemenst, focus_pos)

    for idx in range(num_elemenst):
        karray.add_arc_element(element_pos[:, idx], radius, diameter, focus_pos)

    #grid properties
    N = Vector([256, 256])
    d = Vector([0.5e-3, 0.5e-3])
    kgrid = kWaveGrid(N,d)

    #medium properties
    medium = kWaveMedium(sound_speed=1500)

    #time array
    kgrid.makeTime(medium.sound_speed)

    source = kSource()
    x_offset = 20
    #make a small disc in the top left of the domain
    source.p0 = make_disc(N, Vector([N.x/8+ x_offset, N.y/8]),4)
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
    sensor_data_point = reorder_binary_sensor_data(output["p"].T, reorder_index=reorder_index)

    sensor.mask = karray.get_array_binary_mask(kgrid)

    output = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)
    sensor_data = output["p"].T
    combined_sensor_data = karray.combine_sensor_data(kgrid, sensor_data)



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

    ax2 = fig.add_axes([0.95,0.1,0.03,0.8])
    mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing="proportional", ticks=bounds, boundaries=bounds, format="%1i")

    ax.set_title("Simualtion Layout")
    ax.set_ylabel("Simulation Components[-]", size=12)

    #calculate the middle points for each segment of the colorbar
    mid_points = [(bounds[i] + bounds[i+1])/2 for i in range(len(bounds)-1)]

    #set new tick positions adn labels
    ax2.set_yticks(mid_points)
    ax2.set_yticklabels(list(labels.values()))

    #plot recorded sensor data
    fig, [ax1, ax2] = plt.subplots(ncols=1, nrows=2)
    im1 = ax1.imshow(sensor_data_point, aspect="auto", cmap=get_color_map(), interpolation="none")
    ax1.set_xlabel(r"Time [$\mu$s]")
    ax1.set_ylabel("Detector Number")
    ax1.set_title("Cartesian point detectors")
    fig.colorbar(im1, ax=ax1)

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
    plt.plot(kgrid.t_array.squeeze() * 1e6, sensor_data_point[0, :], label="Cartesian point detectors")
    plt.plot(kgrid.t_array.squeeze() *1e6, combined_sensor_data[0, :], label="Arc detecors")
    plt.xlabel(r"Time [$\mu$s]")
    plt.ylabel("pressure [pa]")
    plt.legend()

    plt.show()



if __name__ == "__main__":
    print("start main")
    main()
    print("main is complteed")